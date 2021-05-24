# 目录

[toc]

---


## 安装 scala

现在 master 机器上安装
```shell

[root@master data]#
[root@master data]# cd packs/
[root@master packs]# ll
总用量 417324
-rw-r--r--. 1 root root 218720521 5月   7 23:28 hadoop-2.7.7.tar.gz
-rw-r--r--. 1 root root 185646832 5月   7 23:28 jdk-8u181-linux-x64.tar.gz
-rw-r--r--. 1 root root  22969424 5月  17 19:50 scala-2.13.4.tgz
[root@master packs]#
[root@master packs]#
[root@master packs]#
[root@master packs]# ll
总用量 417324
-rw-r--r--. 1 root root 218720521 5月   7 23:28 hadoop-2.7.7.tar.gz
-rw-r--r--. 1 root root 185646832 5月   7 23:28 jdk-8u181-linux-x64.tar.gz
-rw-r--r--. 1 root root  22969424 5月  17 19:50 scala-2.13.4.tgz
[root@master packs]#
[root@master packs]#
[root@master packs]#
[root@master packs]# tar zxvf scala-2.13.4.tgz -C /software/
...


[root@master packs]# cd /software/
[root@master software]# ll
总用量 0
drwxr-xr-x. 10 huser ftp  161 5月   9 13:39 hadoop
drwxr-xr-x.  3 root  root  17 5月   9 13:31 hadoop_tmp
drwxr-xr-x.  7    10  143 245 7月   7 2018 java
drwxrwxr-x.  6  2000 2000  79 11月 19 18:29 scala-2.13.4
[root@master software]#



[root@master software]# vim /etc/profile
[root@master software]#

# 追加下面内容到末尾
# scala
export SCALA_HOME=/software/scala-2.13.4
export PATH=$PATH:$SCALA_HOME/bin

[root@master software]# source /etc/profile

[root@master software]# scala -version
Scala code runner version 2.13.4 -- Copyright 2002-2020, LAMP/EPFL and Lightbend, Inc.

[root@master software]#


```



## 安装 spark

```shell

[root@master packs]# pwd
/data/packs
[root@master packs]#
[root@master packs]# ll
总用量 645192
-rw-r--r--. 1 root root 218720521 5月   7 23:28 hadoop-2.7.7.tar.gz
-rw-r--r--. 1 root root 185646832 5月   7 23:28 jdk-8u181-linux-x64.tar.gz
-rw-r--r--. 1 root root  22969424 5月  17 19:50 scala-2.13.4.tgz
-rw-r--r--. 1 root root 233333392 5月  17 19:57 spark-2.4.7-bin-hadoop2.7.tgz
[root@master packs]#

[root@master packs]# tar zxvf spark-2.4.7-bin-hadoop2.7.tgz -C /software/
...


[root@master packs]# cd /software/
[root@master software]#
[root@master software]# ll
总用量 0
drwxr-xr-x. 10 huser ftp   161 5月   9 13:39 hadoop
drwxr-xr-x.  3 root  root   17 5月   9 13:31 hadoop_tmp
drwxr-xr-x.  7    10   143 245 7月   7 2018 java
drwxrwxr-x.  6  2000  2000  79 11月 19 18:29 scala-2.13.4
drwxr-xr-x. 13 huser huser 211 9月   8 2020 spark-2.4.7-bin-hadoop2.7
[root@master software]#
[root@master software]# mv spark-2.4.7-bin-hadoop2.7/ spark-2.4.7
[root@master software]#
[root@master software]# ll
总用量 0
drwxr-xr-x. 10 huser ftp   161 5月   9 13:39 hadoop
drwxr-xr-x.  3 root  root   17 5月   9 13:31 hadoop_tmp
drwxr-xr-x.  7    10   143 245 7月   7 2018 java
drwxrwxr-x.  6  2000  2000  79 11月 19 18:29 scala-2.13.4
drwxr-xr-x. 13 huser huser 211 9月   8 2020 spark-2.4.7
[root@master software]#
[root@master software]#
```

### 配置 spark-env.sh 文件
```shell
[root@master software]# cd spark-2.4.7/conf
[root@master conf]#
[root@master conf]# ll
总用量 36
-rw-r--r--. 1 huser huser  996 9月   8 2020 docker.properties.template
-rw-r--r--. 1 huser huser 1105 9月   8 2020 fairscheduler.xml.template
-rw-r--r--. 1 huser huser 2025 9月   8 2020 log4j.properties.template
-rw-r--r--. 1 huser huser 7801 9月   8 2020 metrics.properties.template
-rw-r--r--. 1 huser huser  865 9月   8 2020 slaves.template
-rw-r--r--. 1 huser huser 1292 9月   8 2020 spark-defaults.conf.template
-rwxr-xr-x. 1 huser huser 4221 9月   8 2020 spark-env.sh.template
[root@master conf]#
[root@master conf]# cp spark-env.sh.template spark-env.sh
[root@master conf]#
[root@master conf]# vim spark-env.sh


# 把SPARK_HOME/conf/下的spark-env.sh.template文件复制为spark-env.sh
[hadoop@hadoop01 apps]$ cd spark-2.2.0/conf
[hadoop@hadoop01 conf]$ mv spark-env.sh.template spark-env.sh


 
# 修改spark-env.sh配置文件，添加如下内容
[hadoop@hadoop01 conf]$ vim spark-env.sh

export JAVA_HOME=/software/java
export HADOOP_CONF_DIR=/software/hadoop/etc/hadoop
export SPARK_MASTER_HOST=master
export SPARK_MASTER_PORT=7077
export SPARK_WORKER_CORES=1
export SPARK_WORKER_MEMORY=1g
 
# 下面是注释
# # 配置JAVA_HOME，一般来说，不配置也可以，但是可能会出现问题，还是配上吧
# export JAVA_HOME=/software/java

# # 一般来说，spark任务有很大可能性需要去HDFS上读取文件，所以配置上
# # 如果说你的spark就读取本地文件，也不需要yarn管理，不用配
# export HADOOP_CONF_DIR=/software/hadoop/etc/hadoop
 
# # 设置 Master 的主机名
# export SPARK_MASTER_HOST=master

# # 提交Application的端口，默认就是这个，万一要改呢，改这里
# export SPARK_MASTER_PORT=7077

# # 每一个Worker最多可以使用的cpu core的个数，我虚拟机就一个...
# # 真实服务器如果有32个，你可以设置为32个
# export SPARK_WORKER_CORES=1

# # 每一个Worker最多可以使用的内存，我的虚拟机就2g
# # 真实服务器如果有128G，你可以设置为100G
# export SPARK_WORKER_MEMORY=1g



[root@master conf]# ll
总用量 44
-rw-r--r--. 1 huser huser  996 9月   8 2020 docker.properties.template
-rw-r--r--. 1 huser huser 1105 9月   8 2020 fairscheduler.xml.template
-rw-r--r--. 1 huser huser 2025 9月   8 2020 log4j.properties.template
-rw-r--r--. 1 huser huser 7801 9月   8 2020 metrics.properties.template
-rw-r--r--. 1 huser huser  865 9月   8 2020 slaves.template
-rw-r--r--. 1 huser huser 1292 9月   8 2020 spark-defaults.conf.template
-rwxr-xr-x. 1 root  root  4425 5月  17 20:07 spark-env.sh
-rwxr-xr-x. 1 huser huser 4221 9月   8 2020 spark-env.sh.template
[root@master conf]#
[root@master conf]# cp slaves.template slaves
[root@master conf]#
[root@master conf]# ll
总用量 48
-rw-r--r--. 1 huser huser  996 9月   8 2020 docker.properties.template
-rw-r--r--. 1 huser huser 1105 9月   8 2020 fairscheduler.xml.template
-rw-r--r--. 1 huser huser 2025 9月   8 2020 log4j.properties.template
-rw-r--r--. 1 huser huser 7801 9月   8 2020 metrics.properties.template
-rw-r--r--. 1 root  root   865 5月  17 20:08 slaves
-rw-r--r--. 1 huser huser  865 9月   8 2020 slaves.template
-rw-r--r--. 1 huser huser 1292 9月   8 2020 spark-defaults.conf.template
-rwxr-xr-x. 1 root  root  4425 5月  17 20:07 spark-env.sh
-rwxr-xr-x. 1 huser huser 4221 9月   8 2020 spark-env.sh.template
[root@master conf]#
[root@master conf]# vim slaves

# 删掉 localhost
# master    如果你要把主机也当做一个 worker 的话就配上主机的 hostname
slave1
slave2

```

## 分发给 slave 节点

```shell

[root@master software]# ll
总用量 0
drwxr-xr-x. 10 huser ftp   161 5月   9 13:39 hadoop
drwxr-xr-x.  3 root  root   17 5月   9 13:31 hadoop_tmp
drwxr-xr-x.  7    10   143 245 7月   7 2018 java
drwxrwxr-x.  6  2000  2000  79 11月 19 18:29 scala-2.13.4
drwxr-xr-x. 13 huser huser 211 9月   8 2020 spark-2.4.7
[root@master software]#
[root@master software]#
[root@master software]# pwd
/software
[root@master software]#
[root@master software]# scp -r spark-2.4.7/ slave1:/software/
...
[root@master software]# scp -r spark-2.4.7/ slave2:/software/
...
```

## 所有节点(master+slaves)  配置 SPARK_HOME 环境变量

```shell
vim ~/.bash_profile

# 注意是直接追加到文件末尾
export SPARK_HOME=/software/spark-2.4.7
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

source ~/.bash_profile
```


## 启动 spark

注意哈，在这一步之前请先确定 hadoop 跟 yarn 的启动状态是正常的


主机：
```shell
[root@master conf]# jps
8016 SecondaryNameNode
8417 Jps
7828 NameNode           # hadoop 正常
8166 ResourceManager    # yarn  正常
```
从机：
```shell
[root@slave1 conf]# jps
13732 DataNode          # hadoop 正常
13911 Jps
13837 NodeManager       # yarn  正常
```
如果不正常的话要重新启动 hadoop


**开始启动 spark**


```shell
[root@master sbin]# pwd
/software/spark-2.4.7/sbin
[root@master sbin]#
[root@master sbin]# ll
总用量 92
-rwxr-xr-x. 1 huser huser 2803 9月   8 2020 slaves.sh
-rwxr-xr-x. 1 huser huser 1429 9月   8 2020 spark-config.sh
-rwxr-xr-x. 1 huser huser 5689 9月   8 2020 spark-daemon.sh
-rwxr-xr-x. 1 huser huser 1262 9月   8 2020 spark-daemons.sh
-rwxr-xr-x. 1 huser huser 1190 9月   8 2020 start-all.sh
-rwxr-xr-x. 1 huser huser 1274 9月   8 2020 start-history-server.sh
-rwxr-xr-x. 1 huser huser 2050 9月   8 2020 start-master.sh
-rwxr-xr-x. 1 huser huser 1877 9月   8 2020 start-mesos-dispatcher.sh
-rwxr-xr-x. 1 huser huser 1423 9月   8 2020 start-mesos-shuffle-service.sh
-rwxr-xr-x. 1 huser huser 1279 9月   8 2020 start-shuffle-service.sh
-rwxr-xr-x. 1 huser huser 3151 9月   8 2020 start-slave.sh
-rwxr-xr-x. 1 huser huser 1527 9月   8 2020 start-slaves.sh
-rwxr-xr-x. 1 huser huser 1857 9月   8 2020 start-thriftserver.sh
-rwxr-xr-x. 1 huser huser 1478 9月   8 2020 stop-all.sh
-rwxr-xr-x. 1 huser huser 1056 9月   8 2020 stop-history-server.sh
-rwxr-xr-x. 1 huser huser 1080 9月   8 2020 stop-master.sh
-rwxr-xr-x. 1 huser huser 1227 9月   8 2020 stop-mesos-dispatcher.sh
-rwxr-xr-x. 1 huser huser 1084 9月   8 2020 stop-mesos-shuffle-service.sh
-rwxr-xr-x. 1 huser huser 1067 9月   8 2020 stop-shuffle-service.sh
-rwxr-xr-x. 1 huser huser 1557 9月   8 2020 stop-slave.sh
-rwxr-xr-x. 1 huser huser 1064 9月   8 2020 stop-slaves.sh
-rwxr-xr-x. 1 huser huser 1066 9月   8 2020 stop-thriftserver.sh
[root@master sbin]#
[root@master sbin]# ./start-all.sh
starting org.apache.spark.deploy.master.Master, logging to /software/spark-2.4.7/logs/spark-root-org.apache.spark.deploy.master.Master-1-master.out
slave2: starting org.apache.spark.deploy.worker.Worker, logging to /software/spark-2.4.7/logs/spark-root-org.apache.spark.deploy.worker.Worker-1-slave2.out
slave1: starting org.apache.spark.deploy.worker.Worker, logging to /software/spark-2.4.7/logs/spark-root-org.apache.spark.deploy.worker.Worker-1-slave1.out
```

启动成功后：
主机：
```shell
[root@master sbin]# jps
8016 SecondaryNameNode
7828 NameNode
8166 ResourceManager
8503 Jps
8442 Master             # spark 正常（主机作为 Master）
[root@master sbin]#
```
从机：
```shell
[root@slave1 conf]# jps
13732 DataNode
13837 NodeManager
13982 Worker            # spark 正常（从机作为 Worker）
14030 Jps
```







### spark-shell 环境

```shell
[root@master spark-2.4.7]# spark-shell
21/05/17 21:25:49 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://master:4040
Spark context available as 'sc' (master = local[*], app id = local-1621257967023).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.7
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_181)
Type in expressions to have them evaluated.
Type :help for more information.

scala> :quit
[root@master spark-2.4.7]#

```








### 测试

先构造数据文件并上传至 hdfs
```shell

[root@master data]# pwd
/code/spark_file/data

[root@master data]# vim sample_test.txt
[root@master data]# hadoop fs -ls /
Found 1 items
drwxr-xr-x   - root supergroup          0 2021-05-09 13:49 /t01
[root@master data]#
[root@master data]#
[root@master data]# hadoop fs -put sample_test.txt /
[root@master data]# hadoop fs -ls /
Found 2 items
-rw-r--r--   3 root supergroup         28 2021-05-17 21:39 /sample_test.txt
drwxr-xr-x   - root supergroup          0 2021-05-09 13:49 /t01
[root@master data]#
[root@master data]#
```


```scala
[root@master spark-2.4.7]# spark-shell

scala>

scala> sc.textFile("/sample_test.txt").flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).collect
res3: Array[(String, Int)] = Array((hello,3), (zsn,1), (world,1))
```



```scala
// 结果保存成文件
scala> sc.textFile("/sample_test.txt").flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).saveAsTextFile("/out.txt")

scala>

```
查看结果：
**注意 
`-get` 指令拉下来的，是一个目录， 
==`-getmerge` 拉下来的才是一个文件==**
```shell
[root@master data]# hadoop fs -get /out.txt /code/spark_file/data/out1.txt
[root@master data]#
[root@master data]# hadoop fs -getmerge /out.txt /code/spark_file/data/out2.txt
[root@master data]#
```
-get 指令拉下来的目录 `out1.txt`
```shell
[root@master data]# pwd
/code/spark_file/data
[root@master data]#
[root@master data]# ll
总用量 4
drwxr-xr-x. 2 root root 40 5月  17 21:49 out1.txt
-rw-r--r--. 1 root root 28 5月  17 21:38 sample_test.txt
[root@master data]#
[root@master data]# cat out1.txt/
cat: out1.txt/: 是一个目录
[root@master data]# cd out1.txt/
[root@master out1.txt]# ll
总用量 4
-rw-r--r--. 1 root root 28 5月  17 21:49 part-00000
-rw-r--r--. 1 root root  0 5月  17 21:49 _SUCCESS
```

-getmerge 拉下来的是文件 `out2.txt`
```shell
[root@master data]# ll
总用量 8
drwxr-xr-x. 2 root root 40 5月  17 21:49 out1.txt       # 目录
-rw-r--r--. 1 root root 28 5月  17 21:51 out2.txt       # 文件
-rw-r--r--. 1 root root 28 5月  17 21:38 sample_test.txt
[root@master data]#
[root@master data]# vim out2.txt
[root@master data]# cat out2.txt
(hello,3)
(zsn,1)
(world,1)
```




---


参考：
