# 目录

[toc]

---

# Spark内存管理之堆内/堆外内存原理详解


[source](https://blog.csdn.net/pre_tender/article/details/101517789)


# 概要

介绍 Spark 内存管理中，涉及到的 ==**堆内内存 (On-heap Memory)**== 和 ==**堆外内存 (Off-heap Memory)**== 的相关原理介绍。

这部分主要在 org.apache.spark.memory.UnifiedMemoryManager.scala 文件中进行描述，关于源码剖析请参考 Spark 内存管理之 UnifiedMemoryManager 。

# 1. 前言
在执行 Spark 的应用程序时， Spark 集群会启动 Driver 和 Executor 两种 JVM 进程，
- **`Driver`** 为主控进程，负责创建 Spark 上下文，提交 Spark 作业（ Job ），并将作业转化为计算任务（ Task ），在各个 Executor 进程间协调任务的调度，
- **`Executor`** 负责在工作节点上执行具体的计算任务，并将结果返回给 Driver ，同时为需要持久化的 RDD 提供存储功能。

由于 Driver 的内存管理相对来说较为简单，本文主要对 Executor 的内存管理进行分析，
==下文中的 **Spark 内存** 均特指 **Executor 的内存**==。
<img width=800 src="https://img-blog.csdnimg.cn/20190927170445802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ByZV90ZW5kZXI=,size_16,color_FFFFFF,t_70#pic_center"></img>

# 2. 堆内和堆外内存
作为一个 JVM 进程， Executor 的内存管理建立在 JVM 的内存管理之上，
**Spark 对 JVM 的堆内（On-heap）空间进行了更为详细的分配**，以充分利用内存。
同时， **==Spark 引入了堆外（Off-heap）内存==，使之可以直接 ==<u>在工作节点的系统内存中开辟空间</u>==**，进一步优化了内存的使用。

<img width=500 src="https://img-blog.csdnimg.cn/2019092717073365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ByZV90ZW5kZXI=,size_16,color_FFFFFF,t_70#pic_center"></img>

## 2.1 堆内内存(On-heap Memory)

堆内内存的大小，由 Spark 应用程序启动时的 **`executor-memory`** 或 **`spark.executor.memory`** 参数配置。 

Executor 内运行的 Task 并发任务 **共享 JVM 堆内内存**，
- 这些 Task 任务在缓存 RDD 和广播（Broadcast）数据时占用的内存被规划为 **存储（Storage）内存**，
- 而这些 Task 任务在执行 Shuffle 时占用的内存被规划为 **执行（Execution）内存**，
- 剩余的部分不做特殊规划，

那些 Spark 内部的对象实例，或者用户定义的 Spark 应用程序中的对象实例，**均占用剩余的空间**。

不同的管理模式下，这三部分占用的空间大小各不相同（下面第 2 小节介绍）。

### 2.1.1 堆内内存的申请与释放
Spark 对堆内内存的管理是一种逻辑上的“规划式”的管理，
**因为对象实例占用内存的申请和释放都由 JVM 完成， Spark 只能在申请后和释放前 ==记录== 这些内存**：
> 申请内存：
Spark 在代码中 new 一个对象实例
JVM 从堆内内存分配空间，创建对象并返回对象引用
Spark 保存该对象的引用，记录该对象占用的内存<br>
释放内存：
Spark 记录该对象释放的内存，删除该对象的引用
等待 JVM 的垃圾回收机制释放该对象占用的堆内内存

### 2.1.2 堆内内存优缺点分析

我们知道，**==堆内内存采用 JVM 来进行管理==**。而 JVM 的对象可以以 **序列化** 的方式存储，
序列化的过程是将对象转换为二进制字节流，
本质上可以理解为将<u>**非连续空间的链式存储**</u>转化为<u>**连续空间或块存储**</u>，

在访问时则需要进行序列化的逆过程 反序列化，将字节流转化为对象，
序列化的方式可以节省存储空间，但增加了存储和读取时候的计算开销。

对于 Spark 中序列化的对象，由于是字节流的形式，其占用的内存大小可直接计算。
对于 Spark 中非序列化的对象，其占用的内存是通过周期性地采样近似估算而得，即并不是每次新增的数据项都会计算一次占用的内存大小。这种方法：

1. 降低了时间开销但是有可能误差较大，导致某一时刻的实际内存有可能远远超出预期
2. 此外，在被 Spark 标记为释放的对象实例，很有可能在实际上并没有被 JVM 回收，导致实际可用的内存小于 Spark 记录的可用内存。
   所以 **==Spark 并不能准确记录实际可用的堆内内存，从而也就无法完全避免内存溢出（OOM, Out of Memory）的异常==**。

虽然不能精准控制堆内内存的申请和释放，
但 Spark 通过 **==对存储 Storage 内存和执行 Execution 内存各自独立的规划管理==**，
可以决定是否要在存储内存里缓存新的 RDD ，以及是否为新的任务分配执行内存，
在一定程度上可以提升内存的利用率，减少异常的出现。

### 2.1.3 堆内内存分区 (静态方式 , 弃)
在静态内存管理机制下，存储内存、执行内存和其他内存三部分的大小在 Spark 应用程序运行期间是固定的，
但用户可以在应用程序启动前进行配置，
堆内内存的分配如图所示：
<img width=900 src="https://img-blog.csdnimg.cn/20190927174923854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ByZV90ZW5kZXI=,size_16,color_FFFFFF,t_70"></img>







<br>
<br><br><br><br><br><br>


<u></u>

<!-- 
<img width=500 src=""></img>
<img style="width:500px" src=""></img>
 -->


<style>
.red {
	color: red;
	font-weight: bold;
}


</style>








