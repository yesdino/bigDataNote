# 目录

[toc]

---

# 服务器

## 服务器基本介绍
[link](https://www.bilibili.com/video/BV1JT4y1g7nM?p=2)
服务器，也称伺服器，是提供计算服务的设备。
由于服务器需要响应服务请求，并进行处理，
因此一般来说服务器应具备承担服务并且保障服务的能力。

在网络环境下，根据服务器提供的服务类型不同，
分为文件服务器、数据库服务器、WEB服务器等。

服务器的构成包括处理器、硬盘、内存、系统总线等，
和通用的计算机架构类似，但是由于需要提供高可靠的服务，
因此在处理能力、稳定性、可靠性、安全性、可扩展性、可管理性等方面要求较高。

可以简单的理解为服务器就是一台电脑，只不过硬盘比普通的PC机更大，
CPU比普通的PC机处理速度更快，网卡比普通的PC机更快。。

服务器特点
- 高处理能力
- 高扩展性
- 高可靠

# Hadoop 框架

## Hadoop

[link](https://www.bilibili.com/video/BV1cW411r7c5?p=9)
Hadoop 是一个由 Apache 基金会所开发的 **分布式系统基础架构**。
主要解决，海量数据的 **存储** 和海量数据的 **分析计算** 问题


广义上来说， ==Hadoop 通常是指一个更广泛的概念== —— Hadoop 生态圈。

![Hadoop生态圈.png](https://upload-images.jianshu.io/upload_images/11876740-d539bc2f8e76be4a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


现在说的 Hadoop 指的是 Hadoop 生态圈里面的一套组件 Storm Spark Redis


[link](https://www.bilibili.com/video/BV1jx411g77S?p=2)
![课程安排.png](https://upload-images.jianshu.io/upload_images/11876740-464c055508d04511.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Hadoop 是什么
[link](https://www.bilibili.com/video/BV1jx411g77S?p=3)
解决问题
- 海量数据的存储（**HDFS**）
- 海量数据的分析（**MapReduce**）(一种分析模型)
- 资源管理调度 （**YARN**）(分发架包，分发不同的资源，然后在不同的机器上给你分配一些资源的容器 )

作者： Doug Cutting
受 Google三篇论文的启发（**GFS**、 **MapReduce**、 **Bigtable**)




### 问题：如何解决海量数据的存储

[00:36](https://www.bilibili.com/video/BV1jx411g77S?p=4)
![问题：如何解决海量数据的存储.png](https://upload-images.jianshu.io/upload_images/11876740-d3ede6eff7ed9218.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**缺点：**
存储是可以的，但是分析会很困难。
系统复杂性、可靠性低

#### HDFS 架构

[03:38](https://www.bilibili.com/video/BV1jx411g77S?p=4)
- **主从结构**
  - 主节点： **`Namenode`**
  - 从节点，有很多个： **`Datanode`**
  
<img width=450 src="https://upload-images.jianshu.io/upload_images/11876740-552a0e0260e8cea3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></img>
- **`Namenode`** (节点) 负责
  - 接收用户操作请求
  - 维护文件系统的目录结构
  - **管理 文件与 block 之间关系， block 与 Datanode之间关系**
- **`Datanode`** (节点) 负责
  - 存储文件
  - 文件被分成 block 存储在磁盘上
  - 为保证数据安全，文件会有多个副本


### 问题：如何解决海量数据的运算

[07:32](https://www.bilibili.com/video/BV1jx411g77S?p=4)


#### MapReduce
<img width=450 src="https://upload-images.jianshu.io/upload_images/11876740-92bf04478c3b3fdd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></img>

- **Map** ：局部逻辑。在 <u>==每个==</u> 局部的节点机器上 **并发** 运行。
- **Reduce** ：只在一台机器上运行，通过网络将其他机器上的 Map 程序运算得到的结果取过来汇总得到结果。



# Linux 服务器准备工作

## 手动分配IP网络地址

[30:50](https://www.bilibili.com/video/BV1jx411g77S?p=5)

不要 DHCP 自动分配 IP 地址，自己手动指定一个 IP 地址。

在公司中的 IP 地址一半都要我们手动定下来，不能让它自动改。



## 将普通用户添加到 sudo 文件中

[39:17](https://www.bilibili.com/video/BV1jx411g77S?p=5)

```html
[hadoop@172 ~]$ sudo vi /etc/inittab

We trust you have received the usual lecture from the local System
Administrator. It usually boils down to these three things:

    #1) Respect the privacy of others.
    #2) Think before you type.
    #3) With great power comes great responsibility.

[sudo] password for hadoop:
hadoop is not in the sudoers file.  This incident will be reported.
```
注意最后一行提示：
`hadoop is not in the sudoers file.  This incident will be reported.`
告诉你 `hadoop` 用户不能使用 `sudo` 指令，因为这个用户没有在 `sudoers` 文件内。


**所以你需要将这个用户添加到 `sudoers` 文件内。**

1、先切换到 `root` 用户下，进入 `/etc/sudoers` 文件：
```html
[hadoop@172 ~]$ su root
Password:
[root@172 hadoop]# vi /etc/sudoers
```

2、找到目标行
```html
## Allow root to run any commands anywhere
root    ALL=(ALL)       ALL
```
3、将要添加的用户添加到下一行
```html
101 hadoop  ALL=(ALL)       ALL
```
4、要记住回到普通用户
```html
[root@172 hadoop]# exit
exit
```

## 修改主机名与 IP 的映射关系

由于后续会部署很多集群，
所以我们需要需要给每一个主机都配置一个与业务相关的主机名，
并且需要将这个主机名与主机的 IP 映射在一起。
使用的时候就直接使用主机名了，不会只用 IP 地址了，这样更方便。

### 修改主机名

[49:43](https://www.bilibili.com/video/BV1jx411g77S?p=5)

```html
[hadoop@172 ~]$ sudo hostname hadoop001
[hadoop@172 ~]$
```

重新登录才能看到效果
```html
[hadoop@172 ~]$ exit
logout
Connection to ip closed.

...

Welcome to Alibaba Cloud Elastic Compute Service !

[hadoop@hadoop001 ~]$
```
看到 `@` 后面跟着的主机名已经发生了改变。



### 修改主机名与 IP 的映射关系

[52:02](https://www.bilibili.com/video/BV1jx411g77S?p=5)

```html
[hadoop@hadoop001 ~]$ sudo vi /etc/hosts
[sudo] password for hadoop:

127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
```
追加下面设置到最后一行
```html
[要映射的IP]  hadoop001
```
然后测试一下主机名能不能映射到 IP

```html
[hadoop@hadoop001 ~]$ ping hadoop001
PING hadoop001 (映射的IP) 56(84) bytes of data.
64 bytes from hadoop001 (映射的IP): icmp_seq=1 ttl=64 time=0.052 ms
64 bytes from hadoop001 (映射的IP): icmp_seq=2 ttl=64 time=0.049 ms
64 bytes from hadoop001 (映射的IP): icmp_seq=3 ttl=64 time=0.051 ms
64 bytes from hadoop001 (映射的IP): icmp_seq=4 ttl=64 time=0.044 ms
^C
--- hadoop001 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 2999ms
rtt min/avg/max/mdev = 0.044/0.049/0.052/0.003 ms
[hadoop@hadoop001 ~]$
```
这样说明映射到了，以后系统会直接将 `hadoop001` 识别为你设定的 IP 地址


## 安装 jdk


### yum 直接安装
1、看一下都有哪些版本
```html
yum -y list java*
```
2、找到自己想要安装的 jdk 包进行安装
```html
[hadoop@hadoop001 ~]$ sudo yum -y install java-1.8.0-openjdk.x86_64
```
3、安装完之后查看 java 指令是否正运行
```html
[hadoop@hadoop001 ~]$ java -version
openjdk version "1.8.0_262"
OpenJDK Runtime Environment (build 1.8.0_262-b10)
OpenJDK 64-Bit Server VM (build 25.262-b10, mixed mode)
[hadoop@hadoop001 ~]$
```

### 下载安装包手动安装

[00:20](https://www.bilibili.com/video/BV1jx411g77S?p=6)

1、上传安装包

2、解压

3、安装 java 到指定目录

4、将 java 添加到环境变量中


## 安装 Hadoop

- 下载
```html
[hadoop@hadoop001 ~]$ wget https://mirror.bit.edu.cn/apache/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz
--2020-09-06 16:47:31--  https://mirror.bit.edu.cn/apache/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz
Resolving mirror.bit.edu.cn (mirror.bit.edu.cn)... 219.143.204.117, 202.204.80.77, 2001:da8:204:1205::22
Connecting to mirror.bit.edu.cn (mirror.bit.edu.cn)|219.143.204.117|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 218720521 (209M) [application/octet-stream]
Saving to: ‘hadoop-2.7.7.tar.gz’

100%[==============================================================================================================================>] 218,720,521 11.9MB/s   in 16s

2020-09-06 16:47:48 (12.8 MB/s) - ‘hadoop-2.7.7.tar.gz’ saved [218720521/218720521]

[hadoop@hadoop001 ~]$ ls
app  hadoop-2.7.7.tar.gz
[hadoop@hadoop001 ~]$

```

- 解压到指定目录
```html
[hadoop@hadoop001 ~]$ tar -zxvf hadoop-2.7.7.tar.gz -C app/
```

- hadoop 目录结构
```html
[hadoop@hadoop001 ~]$ cd app/
[hadoop@hadoop001 app]$ cd hadoop-2.7.7/
[hadoop@hadoop001 hadoop-2.7.7]$ ll
total 136
drwxr-xr-x 2 hadoop hadoop  4096 Jul 19  2018 bin
drwxr-xr-x 3 hadoop hadoop  4096 Jul 19  2018 etc
drwxr-xr-x 2 hadoop hadoop  4096 Jul 19  2018 include
drwxr-xr-x 3 hadoop hadoop  4096 Jul 19  2018 lib
drwxr-xr-x 2 hadoop hadoop  4096 Jul 19  2018 libexec
-rw-r--r-- 1 hadoop hadoop 86424 Jul 19  2018 LICENSE.txt
-rw-r--r-- 1 hadoop hadoop 14978 Jul 19  2018 NOTICE.txt
-rw-r--r-- 1 hadoop hadoop  1366 Jul 19  2018 README.txt
drwxr-xr-x 2 hadoop hadoop  4096 Jul 19  2018 sbin
drwxr-xr-x 4 hadoop hadoop  4096 Jul 19  2018 share
```
hadoop 相关的架包都在 `/` 目录中
```html

```

看到 [15:32](https://www.bilibili.com/video/BV1jx411g77S?p=6)








<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>






