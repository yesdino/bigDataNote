

# 目录

[toc]

---


# 历史介绍

## Hadoop 历史

[02:36](https://www.bilibili.com/video/BV174411X7Pk?from=search&seid=5079076188602324962)

Hadoop 历史
2003，2004 Google 2篇论文
- 2011 年发布 1.0 版本
- 2012 年发布稳定版
- 2013 年发布 2.X 版


### 1.0 版本

[08:23](https://www.bilibili.com/video/BV174411X7Pk?from=search&seid=5079076188602324962)

早期版本的架构：
<img width='650' src='https://upload-images.jianshu.io/upload_images/11876740-73d28b3313bb10a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>


**1.0 MR 的缺点：**
mr 基于数据集的计算，所以面向数据
1. 基本运算规则从存储介质中获取（采集）数据，然后进行计算，最后将结果存储到介质中，所以主要应用于 **一次性计算**，不适合于数据挖掘和机器学习这样的送代计算和图形挖掘计算。
2. MR 基于文件存储介质的操作，所以性能非常的慢
3. MR 和 hadoop 紧空耦合在一起


### 2.X 版本（Yarn）

[16:38](https://www.bilibili.com/video/BV174411X7Pk?from=search&seid=5079076188602324962)

<img width='650' src='https://upload-images.jianshu.io/upload_images/11876740-6cb40fc474bcefc3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

实现了 **资源(ResourceManager)** 与 **任务(Driver)** 的解耦合，
AM(ApplicationMaster) 作为中间层，负责资源与任务的中间交互



## Spark 历史
[26:47](https://www.bilibili.com/video/BV174411X7Pk?from=search&seid=5079076188602324962)

2013 年 6 月发布
Spark 基于 Hadoop1.X 架构思想，采用自己的方式改善 Hadoop1.x 中的 题
Spark 计算基于内存，并且**基于 scala 语法开发**，所以**天生适合迭代式计算**

<img width='450' src='https://upload-images.jianshu.io/upload_images/11876740-a48049b70ece954a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>


**==spark 与 Hadoop，yarn, HDFS 的关系==**：[34:20](https://www.bilibili.com/video/BV174411X7Pk?from=search&seid=5079076188602324962)
实际上面的 spark 架构只是计算架构，不包括存储架构，所以还需要搭配使用 Hadoop 的 HDFS 文件系统作为存储架构。
所以在 Hadoop 的架构中 yarn 支持计算架构的插拔，可以兼容 spark。
所以如果搭配 HDFS 的话，会变成这样：
- spark + yarn 作为计算
- HDFS 作为存储



## spark 内置模块

[00:50](https://www.bilibili.com/video/BV174411X7Pk?p=2)
<img width='600' src='https://upload-images.jianshu.io/upload_images/11876740-effd07553811d932.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>



## spark 特点

1) **快**：与 Hadoop 的 Mapreduce 比，Spark 基于内存的运算要快 100 倍人上，基于硬盘的运算也要快 10 倍以上。 
   Spark 实现了高效的 DAG 执行引擎，可以通过基于内存来高效处理踞流。
   计算的中间结果是存在于内存中的
1) **易用**： Spark 支持 Jawa 、 Python 和 sclca 的 API ，还支持超过 8 种高级算法，使用户可以快速构建不同的应用。
   而且 Spark 支持交互式的 Python 和 Scala 的 Shell ，可以非常方便地在这些 Shel 中使用 Spark 集群来验证解决问题的方法。
1) **通用**： Spark 提供了统一的解决方案。 
   Spark 可以用于批处理、交互式查询 (Spark SQL)、实时流处理 ( Spark Streaming)、机器学习 (Spark Millib)和图计算 ( Graphx)。这些不同类型的处理都可以在同个应用中无缝使用。減少了开发和维护的人力成本和部署平台的物力成本。
2) **兼容性**： Spark 可以非常方便地与其他的开源产品进行融合。
   比如， Spark 可以使用 Hadoop 的 YARN 和 Apache Mesos 作为它的资源管理和调度器，并且可以处理所有 Hadoop 支持的效据，包括 HDFS 、 Hbase 等。
   这对于已经部署 Hadoop 集群的用户特别重要，因为不需要做任何数据迁移就可以使用 Spark 虽大处理能的。


# 重要角色
[00:57](https://www.bilibili.com/video/BV174411X7Pk?p=3)

## 实际上的 spark 架构


<img width='600' src='https://upload-images.jianshu.io/upload_images/11876740-7a34eea72a629e3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

由于 spark 是单单的计算架构，所以
调度：<b style="color:red">Application</b>
计算：<b style="color:blue">Driver-Executor (其中 Driver 是做管理任务的，Executor 是真正做计算的)</b>

<b style="color:green">Master-Worker</b> 则可能会没有这一层（如果在 yarn 中的话就没有这层）



## Driver (驱动器)

Spark 的驱动器是执行开发程序中的 main 方法的进程。
它负责开发人员编写的用来创建 Spark Context 、创建 RDD ，以及进行 RDD 的转化操作和行动操作代码的执行。
如果你是用 spark shell ，那么当你启动 Spark shell 的时候，系统后台自启了一个 Spark 驱动器程序，就是在 Spark shell 中预加载的**一个叫作 sc 的 Spark Context 对象**。
如果驱动器程序终止，那么 Spark 应用也就结東了。
**主要负责**：
1) 把用户程序转为作业（Job)
2) 跟踪 Executor 的运行状况
3) 为执行器节点调度任务 
4) UI 展示应用运行状况






## Executor（执行器） 

Spark Executor 是一个工作进程，负责在 Spark 作业中运行任务，**任务间相互独立**。 
Spark 应用启动时，Executor 节点被同时启动，并且始终伴随着整个 Spark 应用的生命周期而存在。
<u>如果有 Executor 节点发生了故障或崩溃， Spark 应用也可以继续执行，会将出错节点上的任务调度到其他 Executor 节点上继续运行。</u>
**主要负责**：
1) 负责运行组成 Spark 应用的任务，并将结果返回给驱动器进程
2) 通过自身的块管理器（Block Manager）**为用户程序中要求缓存的 RDD 提供内存式存储**。 
   RDD 是直接缓存在 Executor 进程内的，因此任务可以在运行时充分利用缓存数据加速运算



# Local 模式
Local 模式就是运行在一台计算机上的模式，通常就是用于在本机上练手和測试。它可以通过以下集中方式设置 Mastel

- **`local`** ：所有计算都运行在一个线程当中，没有任何并行计算，通常我们在本机执行一些测測试代码，或者练手，就用这种模式：
- **`local[K]`** : 指定使用几个线程米运行计算，比如 `local[4]` 就是运行 4 个 Worker 线程。
  **==通常我们的 Cpu 有几个 Core==**(本地模式时) ，就指定几个线程，最大化利用 Cpu 的计算能力
- **`local[*]`** ： 这种模式直接帮你按照 Cpu 最多 Cores 来设置线程数了。




# 安装使用

[03:19](https://www.bilibili.com/video/BV174411X7Pk?p=4)

**demo**:
1) 上传并解压 spark 安装包
<img width='600' src='https://upload-images.jianshu.io/upload_images/11876740-889ceac9e824fa91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

2) 官方求 PI demo
<img width='500' src='https://upload-images.jianshu.io/upload_images/11876740-afc4bc7577827223.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>


## 基本语法，参数说明

[01:33](https://www.bilibili.com/video/BV174411X7Pk?p=5)

**基本语法：**
```shell
bin/spark-submit\
--class <main-class> \  # 执行 jar 包中的那个类
--master <master-url> \
--deploy-mode <deploy-mode> \
--conf<kev>=<value> \
...# other options
<application-jar> \     # jar 包
[application-arguments]
```

**参数说明：**
- **`--master`** 指定 Master的地址，默认为 Local
- **`--class`**：你的应用的启动类（如org. apache. spark, examples. Spark Pi
- **`--deploy-mode`**：是否发布你的驱动到 worker节点（ cluster）或者作为一个本地客户端（client)(default: client)*.
- **`--conf`**：任意的 Spark配置属性，格式 key=value。如果值包含空格，可以加引号 "key-value"
- **`application-jar`**：打包好的应用ja，包含依赖。这个 URL 在集群中全局可见。比如hds：∥共享存储系统，如果是file:/path，那么所有的节点的 path 都包含同样的janr
- **`application-arguments`**：传给 main() 方法的参数
- **`--executor-memory 1G`** 指定每个 executor可用内存为1G
- **`--total- executor-cores 2`** 指定每个 executor使用的cup核数为2个



## **WordCount**

[02:44](https://www.bilibili.com/video/BV174411X7Pk?p=5)

读取文件

### 思路逻辑
<img width='600' src='https://upload-images.jianshu.io/upload_images/11876740-a66ee786faae8127.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>
<br>

### scala 代码实现

[link](https://www.bilibili.com/video/BV174411X7Pk?p=6)
<img width='800' src='https://upload-images.jianshu.io/upload_images/11876740-cad2ebb4e054b1da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>
```scala
scala> sc.textFile("input").flatMap(_.split(" ").map((_,1)).reduceByKey(_+_). collect
res3: Array[(String, Int)] Array((Spark, 1), (World, 1),(Scala, 1), (Hello, 3))
```

### 数据流分析
[05:53](https://www.bilibili.com/video/BV174411X7Pk?p=7)
- **`textFile("input")`**：读取本地文件 inpt 文件夹数据；
- **`flatmap(spit(""))`**: 压平操作，按照空格分割符将一行数据映射成一个个单词 
- **`map((_,1))`** : 对每一个元素操作，将单词映射为元组 
- **`reduceByKey(_+_)`**: 按照 key 将值进行聚合，相加； 
- **`collect`** ：将数据收集到 Driver 端展示。

<img width='900' src='https://upload-images.jianshu.io/upload_images/11876740-0967b1b923b47b62.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

<br>
<br>
<br>

**Spark 通用运行简易流程**
[01:50](https://www.bilibili.com/video/BV174411X7Pk?p=7)
<img width='700' src='https://upload-images.jianshu.io/upload_images/11876740-574dc31fd0b8337e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>




### IDEA 开发

[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=8)

<img width='750' src='https://upload-images.jianshu.io/upload_images/11876740-3a821731f5d54b04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>


# **Yarn 模式**（spark 部署在 yarn

[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=9)

**Spark 客户端直接连接 Yarn** ，不需要额外构建 Spark 集群。
有 yarn-client 和 yarn-cluster 两种模式，
<u>主要区别在于： **Driver 程序的运行节点**。</u>
- **`yarn-client`**  : Driver 程序运行在客户端，适用于交互、调试，**希望立即看到 app 的输出** 
- **`yarn-cluster`**: Driver 程序运行在由 RM(Resourcemanager）启动的 AP(APPMaster）适
用于生产环境。

## 安装使用

1、修改 Hadoop。在 `/hadoop/etc/hadoop` 目录下的配置文件 **`yarn-site.xml`**，添加如下内容：
<img width='800' src='https://upload-images.jianshu.io/upload_images/11876740-62d8c49e781a8c3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>


2、添加如下配置文件配置，在 `spark/conf` 目录下修改 **`spark-env.sh`**(直接修改`.template` 模板文件)，：
```html
YARN_CONF_DIR=/opt/module/hadoop/etc/hadoop
```

<img width='700' src='https://upload-images.jianshu.io/upload_images/11876740-d1697055e8ff13d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

4、执行一个程序
```html
[atguiguehadoop102 spark]$ bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
-master yarn \
-deploy-mode client \
/examples/jars/spark-examples_2.11-2.1.1.jar \
100
```
<b style="color:red;">注意：在提交任务之前需启动 HDFS 以及 YARN 集群。</b>
(因为 yarn 需要 HDFS 的文件系统，不开启来是会出问题的)

## 日志查看

spark 的执行日志是无法直接查看的，因为部署架构是 spark 做计算，yarn 做资源调度，
意味着 spark 与 yarn 没有直接关系，只是靠中间的转换让 spark 部署在 yarn 中执行，所以中间的日志无法直接查看。
得做相关配置让 yarn 看到 spark 的相关执行日志

- 配置 [08:57](https://www.bilibili.com/video/BV174411X7Pk?p=9)
<img width='650' src='https://upload-images.jianshu.io/upload_images/11876740-90101f3ac6e6f193.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

[13:](https://www.bilibili.com/video/BV174411X7Pk?p=9)


## **spark 在 yarn 中的运行原理**

[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=10)

### yarn 架构执行原理
[03:43](https://www.bilibili.com/video/BV174411X7Pk?p=10)

<img width='' src='https://upload-images.jianshu.io/upload_images/11876740-ac6bb4409d8d828d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

### spark 在 yarn 中执行原理
注意：**==spark 是关联在 NodeManager 中的 Container 中的==**

[详解 07:43](https://www.bilibili.com/video/BV174411X7Pk?p=10)。 [快速概括 00:00](https://www.bilibili.com/video/BV174411X7Pk?p=11)
<img width='' src='https://upload-images.jianshu.io/upload_images/11876740-6e0fc2ea49dc07a1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

<img width='' src='https://upload-images.jianshu.io/upload_images/11876740-a05da0781be12a89.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>



## 

[02:47]()





# 将 spark 部署到集群（**TODO**）
[P9 | 将 spark 部署到集群](https://www.bilibili.com/video/BV174411X7Pk?p=9) 
**看了没整理笔记**





<br><br><br>

# yarn 部署 spark 流程图（**TODO**）
[P10 | yarn 部署 spark 流程图](https://www.bilibili.com/video/BV174411X7Pk?p=10) 
**看了 没整理笔记**






<br><br><br>

# 将开发的程序打包到正式环境中运行（**TODO**）
[P11 | 将开发的程序打包到正式环境中运行](https://www.bilibili.com/video/BV174411X7Pk?p=11) 
**看了 没整理笔记**






<br><br><br>

# 本地环境调试
[P12 | 本地环境调试](https://www.bilibili.com/video/BV174411X7Pk?p=12) 

本地 Spark 程序调试需要使用 local 提交模式，(==**意思是只有 local 模式才能进行调试**==)
即将本机当做运行环境，Master 和 Worker 都为本机。
运行时直接加断点调试即可。

如下创建 SparkConf 的时候设置额外属性，表明本地执行：
**`val conf = new SparkConf().setAppName(WC).setMaster("local[*]")`**

## windows bin winutils Error
[00:42](https://www.bilibili.com/video/BV174411X7Pk?p=12) 
如果本机操作系统是 windows ，如果在程序中使用了 hadoop 相关的东西，
比如写入文件到 HIDFS ，则会遇到如下异常：
```
2017-09-14 16: 08: 34 907 ERROR [main] org.apache.hadoop.util.Shell(line:303):
Failed to locate the winutils binary in the hadoop binary path java. io IOException: 
Could not locate executable null\bin winutils.exe in the Hadoop binaries.
```
<img width='800' src='https://upload-images.jianshu.io/upload_images/11876740-0a269d84dd6427f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>



<br><br><br>

# Standalone 独立部署方式
[P13 | 独立部署方式](https://www.bilibili.com/video/BV174411X7Pk?p=13)

独立部署方式，既不用 yarn，就只用 spark 自己。
那么少了 yarn 的资源调度，意味着没有了 ResourceManager(RM) 和 NodeManager(NM) 了,
所以调度资源就会响应的变成 Master 和 Worker
- Master 对应 ResourceManager
- Worker 对应 NodeManager



<br>


[01:18 部署 standalone 独立部署方式](https://www.bilibili.com/video/BV174411X7Pk?p=13)






<br><br>
<br>

# java io 回顾（**TODO**）
[P14 | java io 回顾](https://www.bilibili.com/video/BV174411X7Pk?p=14) **没看**






















# RDD
[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=15)

RDD 是一种 **装饰者设计模式**
<img width='800' src='https://upload-images.jianshu.io/upload_images/11876740-f4a9161d6cef92c4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

一层一层包装的目的，不是为了直接把数据读过来，
而是通过 **==在每一层转换数据的结构==** 的方式，达到统计的逻辑

## 装饰者设计模式
[10:34](https://www.bilibili.com/video/BV174411X7Pk?p=15)
RDD 只有在 collect() 触发之后才会真正开始去 WordFile 读数据，
而前面 collect 之前的一层一层其实是在数据到达之前先封装逻辑
所有 **==RDD 实际上是将数据处理逻辑进行封装==**，类似于装饰器生成器的逻辑，将会大大加快数据处理速度。



## 1 RDD 概述

### 1.1 什么是 RDDS
[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=16)

RDD (**Resilient Distributed Dataset**）做 **弹性分布式数据集**，
是 Spark 中最基本的数据(计算逻辑)抽象。
代码中是一个抽象类，它代表一个不可变、可分区、里面的元素可并行计算的集合。

解释：
**分布式**：数据来源可以来自各个节点，不管来源，都可以一起读取
**数据集**：把多个数据来源的数据采集到一起形成一个集合，即为数据集
**计算逻辑抽象**：将每一个处理数据的逻辑过程抽象成一个 RDD
**不可变**：RDD 声明之后不可变
**可分区，可并行计算**：数据存储在不同的分区，运行时跑在不同机器上的 Executor，即并行


### 1.2 RDD 的属性


1) 一组分区(Partition)，即数据集的基本组成单位
2) 一个计算每个分区的函数；
3) RDD 之间的依赖关系（A 用到了 B，既 A 依赖 B）。依赖关系又称为 “**血缘**” or “血统”
4) 一个 Partitioner, 即 RDD 的分片函数
5) 一个列表，存储存取每个 Partition 的 **优先位置**( prefered location)
**==移动数据不如移动计算==** [03:10](https://www.bilibili.com/video/BV174411X7Pk?p=17)







### 1.3 RDD 特点

[08:30](https://www.bilibili.com/video/BV174411X7Pk?p=17)

RDD 表示 **只读** 的分区的数据集，
对 RDD 进行改动，只能通过 RDD 的转换操作，由一个 RDD 得到一个新的 RDD, 
新的 RDD 包含了从其他 RDD 衍生所必需的信息。 

RDDs 之间存在依赖， RDD 的执行是按照血缘关系延时计算的。
如果血缘关系较长，可以通过持久RDD 来切断血缘关系。



#### 1.3.1 分区
[09:32](https://www.bilibili.com/video/BV174411X7Pk?p=17)

RDD 逻辑上是分区的，每个分区的数据是抽象存在的，计算的时候会通过一个 compute 函数得到每个分区的数据。
如果 RDD 是通过已有的文件系统构建，则 compute 函数是读取指定文件系统中的数据，
如果 RDD 是通过其他 RDD 转换而来，则 compute 函数是执行转换逻辑将其他 RDD 的数据进行转换。

RDD 的分区是为什么了能够 **并行计算**

<br>

#### 1.3.2 只读
[09:47](https://www.bilibili.com/video/BV174411X7Pk?p=17&t=9m47s)
RDD 是只读的，要想改变 RDD 中的数据，只能在现有的 RDD 基础上创建新的 RDD 。

**由一个 RDD 转换到另一个 RDD, 可以通过丰富的操作算子实现**，
不再像 Mapreduce 那样只能写 map 和 reduce 了，如下图所示。

[10:05 什么是算子](https://www.bilibili.com/video/BV174411X7Pk?p=17)
算子：从认知心理学角度，解决问题其实是将问题的初始状态，通过一系列的操作 (Operate) （算子）对可是题的状态进行转换，然后达到完成（解决）状态 
（**==其实算子(operator)就是操作，在 spark 中就是方法==**）

Spark 中的所有的 RDD 方法都称之为算子，但是分为 2 大类：**转换算子** & **行动算子**
- **转换算子**：转换数据结构
- **行动算子**：真正开始数据处理

<br>

#### 1.3.3 依赖
[15:44](https://www.bilibili.com/video/BV174411X7Pk?p=17)
这里直接跳过了，说后面再详细讲

<br>

#### 1.3.4 缓存
[15:56](https://www.bilibili.com/video/BV174411X7Pk?p=17)
为什么防止依赖关系在后面数据处理时发生问题导致中断，找不到前面的依赖关系。
所以将血缘关系缓存起来，防止数据的丢失、血缘的中断

相当于生活中的族谱，方便能找到你的祖先

---






































