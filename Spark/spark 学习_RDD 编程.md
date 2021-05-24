# 目录

[toc]

---




# 2. RDD 编程

[00:00](https://www.bilibili.com/video/BV174411X7Pk?p=18)



## 2.2 RDD 的创建。

在 Spark 中创建 RDD 的创建方式可以分为三种：
- 1、从集合中创建 RDD
- 2、从外部存储创建 RDD
- 3、从其他 RDD 创建

### 2.2.1 从集合中创建

[00:55](https://www.bilibili.com/video/BV174411X7Pk?p=18)
从集合中创建 RDD, 其实就是 **从内存中创建 RDD**
Spark 主要提供了两种函数： **`parallelize`** 和 **`MakeRDD`**

- **①、使用 `parallelize` 从集合创建**  (parallelize 是并行的意思)
```scala
val rdd = sc parallelize(Array(1,2,3,4,5,6,7,8))
// rdd: org.apache.spark.rdd.RDD[Int]=Parallelcollectionrdd[0] at parallelize at <console>: 24
```

- **②、使用 `MakeRDD` 从集合创建**
```scala
val rdd1 = sc.MakeRDD(Array(1,2,3,4,5,6,7,8))
// rdd1: org.apache.spark.rdd.RDD[Int]=ParallelCollectionrdd[1] at MakeRDD at <console>: 24
```
两种方式的底层都是用 `parallelize` 来实现的，所以用哪一种随你喜欢

### 2.2.2 由外部存储系统的数据集创建

[08:40](https://www.bilibili.com/video/BV174411X7Pk?p=18)
包括本地的文件系统，还有所有 Hadoop支持的数据集，比如 HDFS、 Cassandra、 Hbase等，
我们会在第4章详细介绍。

```scala
val rdd2 = sc.textfile("hdfs://hadoop102:9000/RELEASE")
// rdd2: org.apache.spark.rdd.RDD[string] = hdfs://hadoop102: 9000/RELEASE
// MapPartitionsRDD[4] at textFile at <console>: 24
```
注意：
上面代码中的路径，
- 默认情祝下，可以读取项目的本地路径，
- 也可以读取其他路径，比如上面的 HDFS 路径，这个时候路径就要写 HDFS 的路径了

默认从文件中读取的数据都是 **字符串类型**


### 2.2.3 从其他 RDD 创建

详见 2.3 节




## 分区

[看到 00:00](https://www.bilibili.com/video/BV174411X7Pk?p=19)







<br>
<br><br><br><br><br><br><br>

<img width='' src=''>

<u></u>

