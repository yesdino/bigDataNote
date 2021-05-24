# 目录

[toc]

---

## 1. 本地写好 scala 程序

```scala
object WordCount {
    def main(arg:Array[String]) : Unit = {
        val sparkConf = new SparkConf().setAppName("WordCount")
        val sc = new SparkContext(sparkConf)
        sc.textFile("/words.txt").flatMap(_.split(" ")).map.((_,1)).reduceByKey(_+_).saveAsTextFile("/out")
        sc.stop
    }
}
```
把上面的程序打包成 **`wc.jar`** 准备 commit 到集群上面去运行



## 2、提交指令

[link](https://www.bilibili.com/video/BV1BK4y1x7Z5/?spm_id_from=333.788.recommend_more_video.0&t=5m27s)

```shell
spark-submit --master spark://node1:7077 --class WordCount --executor-memory 1g --total-executor-core 4 wc.jar
```
指令解析：
**`--master spark://node1:7077`** ：提交到集群中的某个 node 节点去运行，后面跟着提交的 node 节点的 IP
**`--class WordCount`**           ：jar 包中要执行的类
**`--executor-memory 1g`**        ：指定每个 executor 内存的大小
**`--total-executor-core 4`**     ：指定总的 executor CPU 核数
**`wc.jar`**                      ：执行当前要执行的程序所在的 jar 包




## Driver 端工作

(1) Driver 端会运行客户端程序中的 main 方法

(2) 在 main 方法中构建了 SparkContext 对象，该对象非常重要，它是所有 spark 程序的执行入口
在构建 Spark 对象的内部，也构建了这 2 个对象，一个是 DAGScheduler, 一个是 TaskScheduler

(3) 程序中涉及到了 rdd 大量的转換操作，最后给定一个 action, 触发任务的真正运行，这里就先按照 rdd 与 rdd 之间的依赖
关系，先生成了一张 DAG 有向无环图，图的方向就是 rdd 的算子操作顺序。最后是把 DAG 有向无环图发送给
DAGScheduler 对象

(4) DAGScheduler 获取得到 DAG 有向无环图之后，按照宽依赖，进行 stage 的划分，由于 rdd 的算子操作中，存在大量
的宽依赖，这里划分出了很多个 stage, 每一个 stage 内部有很多可以并行运行的 task 线程，最后是把这些并行运行的 task
线程封装在一个 tasksetf 集合中。把一个一个的 taskSet 集合发送给 TaskScheduler 对象

(5) TaskScheduler 对象获取得到了这些 taskSet 集合之后，然后 stage 与 stage 之间的依赖关系，前面 stager 中 task 先运
行，后面 stage 中的 task 在运行，最后 TaskScheduler 对象依次遍历每个 taskSet 集合，取出每一个 task, 最后提交每
一个 task 到 worker 节点上的 executor 进程中去运行。












