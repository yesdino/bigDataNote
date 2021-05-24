# 目录

[toc]

---

在 VMware 虚拟机中查看自己的 ip 前缀：
NAT 服务 **`192.168.64.*`**



```html
<!-- 主机名:  IP -->
node01: 192.168.64.80   (NN, DN)
node02: 192.168.64.91   (DN)
node03: 192.168.64.92   (DN, SN)
```

NN - NameNode
DN - DataNode
SN - 持久化 NameNode


镜像文件存放磁盘

默认：
```shell
C:\Users\zhang\Documents\Virtual Machines\CentOS 7 64 位
```
| 虚拟机镜像文件路径                 | 虚拟机名称 | 机器 hostname |
| :--------------------------------- | :--------- | :------------ |
| E:\VMwareFile\VirtualMachines\hd01 | node01     | master        |
| E:\VMwareFile\VirtualMachines\hd02 | node02     | slave1        |
| E:\VMwareFile\VirtualMachines\hd03 | node03     | slave2        |

## 安装基本工具
```shell
# 安装网络工具
sudo yum -y install net-tools
# 安装 vim
sudo yum -y install vim


# sudo rpm -ivh jdk-8u112-linux-x64.tar.gz
```


## 修改机器名称

```shell
vi /etc/hostname
```

## 修改 IP 映射

在 master 中修改
```shell
vi /etc/hosts

192.168.64.80    master
192.168.64.91    slave1
192.168.64.92    slave2
```

远程拷贝到 slave 机器



## 关闭防火墙

所有机器都要执行下面的操作

`systemctl stop firewalld` 关闭防火墙 `chkconfig iptables off` 永久关闭防火墙
```shell
systemctl stop firewalld.service
systemctl disable firewalld.service
```
完了查看一下防火墙的状态
```shell
systemctl status firewalld.service
```

## 关闭 selinux

所有机器都要执行下面的操作

```shell
vi /etc/selinux/config
# 文件中的 SELINUX=enforcing 修改为 SELINUX=disabled
```

## 配置免密

注意这一步在 master 机器上执行

生成公钥
```shell
[root@master ~]#
[root@master ~]# cd
[root@master ~]#
[root@master ~]# ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (/root/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /root/.ssh/id_rsa.
Your public key has been saved in /root/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:c70OBV1VVE8ND+gI9i5S8I/TjCNCDlgBWMYhorwr+tc root@master
The key's randomart image is:
+---[RSA 2048]----+
|=+=o         .++X|
|=oo   . o  ... +o|
|.+     + o.o.   o|
|. o .   o oo.    |
| . +   .SB. o    |
|  . o o *o=. .   |
|..   o o +. .    |
|o   . E    o     |
|....        .    |
+----[SHA256]-----+
[root@master ~]#
[root@master ~]#
```
生成公钥私钥
```shell
[root@master ~]# ls -a
.  ..  anaconda-ks.cfg  .bash_history  .bash_logout  .bash_profile  .bashrc  .cshrc  .oracle_jre_usage  .ssh  .tcshrc  .viminfo
[root@master ~]#
[root@master ~]# cd .ssh/
[root@master .ssh]#
[root@master .ssh]# ls
id_rsa  id_rsa.pub  known_hosts
[root@master .ssh]#
```
给本机配置免密
```shell
[root@master .ssh]# ssh-copy-id localhost
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/root/.ssh/id_rsa.pub"
The authenticity of host 'localhost (::1)' can't be established.
ECDSA key fingerprint is SHA256:64n0mG4pDvFJvVUUyCHb5zNc1OqMflsj+MLvtqktW74.
ECDSA key fingerprint is MD5:fc:23:a0:42:5d:82:17:4c:93:b6:7d:cc:c2:3b:48:c7.
Are you sure you want to continue connecting (yes/no)? yes
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
root@localhost's password:

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'localhost'"
and check to make sure that only the key(s) you wanted were added.

[root@master .ssh]# ls
authorized_keys  id_rsa  id_rsa.pub  known_hosts
```
给其他 slave 机器配置免密
```shell
[root@master .ssh]# ssh-copy-id slave1
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/root/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
root@slave1's password:

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'slave1'"
and check to make sure that only the key(s) you wanted were added.

```

```shell
[root@master .ssh]# ssh-copy-id slave2
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/root/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
root@slave2's password:

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'slave2'"
and check to make sure that only the key(s) you wanted were added.

```
测试免密是否生效
```shell
[root@master .ssh]# ssh slave1
Last login: Sun May  9 11:32:02 2021 from 192.168.64.1
[root@slave1 ~]# 登出
Connection to slave1 closed.
[root@master .ssh]# ssh slave2
Last login: Sun May  9 11:32:18 2021 from 192.168.64.1
[root@slave2 ~]# 登出
Connection to slave2 closed.
[root@master .ssh]#

```


## 解压 Java 和 Hadoop 

注意这步先在 master 机器上执行

```shell

[root@master ~]# mkdir -p /data/packs
[root@master ~]# mkdir /software


[root@master /]# ll /data/packs/
总用量 394892
-rw-r--r--. 1 root root 218720521 5月   7 23:28 hadoop-2.7.7.tar.gz
-rw-r--r--. 1 root root 185646832 5月   7 23:28 jdk-8u181-linux-x64.tar.gz
[root@master /]#
```

```shell
[root@master /]# tar zxvf /data/packs/jdk-8u181-linux-x64.tar.gz -C /software/

[root@master /]# tar zxvf /data/packs/hadoop-2.7.7.tar.gz -C /software/
```


```shell
[root@master /]# mv /software/jdk1.8.0_181/ /software/java
[root@master /]# mv /software/hadoop-2.7.7/ /software/hadoop
[root@master /]#
```


## 配置环境变量

注意这步先在 master 机器上执行
```shell
[root@master /]# vim /etc/profile

# java and hadoop
export JAVA_HOME=/software/java
export HADOOP_HOME=/software/hadoop
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin
```
使环境变量生效
```shell
[root@master /]# . /etc/profile
[root@master /]#
[root@master /]#
```
测试环境变量是否生效
```shell
[root@master /]# java -version
java version "1.8.0_181"
Java(TM) SE Runtime Environment (build 1.8.0_181-b13)
Java HotSpot(TM) 64-Bit Server VM (build 25.181-b13, mixed mode)
[root@master /]#
[root@master /]# hadoop version
Hadoop 2.7.7
Subversion Unknown -r c1aad84bd27cd79c3d1a7dd58202a8c3ee1ed3ac
Compiled by stevel on 2018-07-18T22:47Z
Compiled with protoc 2.5.0
From source with checksum 792e15d20b12c74bd6f19a1fb886490
This command was run using /software/hadoop/share/hadoop/common/hadoop-common-2.7.7.jar
[root@master /]#
```


## 修改 hadoop 配置文件

```shell
[root@master hadoop]# pwd
/software/hadoop/etc/hadoop
[root@master hadoop]#
[root@master hadoop]#
[root@master hadoop]# ll
总用量 152
-rw-r--r--. 1 huser ftp  4436 7月  19 2018 capacity-scheduler.xml
-rw-r--r--. 1 huser ftp  1335 7月  19 2018 configuration.xsl
-rw-r--r--. 1 huser ftp   318 7月  19 2018 container-executor.cfg
-rw-r--r--. 1 huser ftp   774 7月  19 2018 core-site.xml
-rw-r--r--. 1 huser ftp  3670 7月  19 2018 hadoop-env.cmd
-rw-r--r--. 1 huser ftp  4224 7月  19 2018 hadoop-env.sh
-rw-r--r--. 1 huser ftp  2598 7月  19 2018 hadoop-metrics2.properties
-rw-r--r--. 1 huser ftp  2490 7月  19 2018 hadoop-metrics.properties
-rw-r--r--. 1 huser ftp  9683 7月  19 2018 hadoop-policy.xml
-rw-r--r--. 1 huser ftp   775 7月  19 2018 hdfs-site.xml
-rw-r--r--. 1 huser ftp  1449 7月  19 2018 httpfs-env.sh
-rw-r--r--. 1 huser ftp  1657 7月  19 2018 httpfs-log4j.properties
-rw-r--r--. 1 huser ftp    21 7月  19 2018 httpfs-signature.secret
-rw-r--r--. 1 huser ftp   620 7月  19 2018 httpfs-site.xml
-rw-r--r--. 1 huser ftp  3518 7月  19 2018 kms-acls.xml
-rw-r--r--. 1 huser ftp  1527 7月  19 2018 kms-env.sh
-rw-r--r--. 1 huser ftp  1631 7月  19 2018 kms-log4j.properties
-rw-r--r--. 1 huser ftp  5540 7月  19 2018 kms-site.xml
-rw-r--r--. 1 huser ftp 11801 7月  19 2018 log4j.properties
-rw-r--r--. 1 huser ftp   951 7月  19 2018 mapred-env.cmd
-rw-r--r--. 1 huser ftp  1383 7月  19 2018 mapred-env.sh
-rw-r--r--. 1 huser ftp  4113 7月  19 2018 mapred-queues.xml.template
-rw-r--r--. 1 huser ftp   758 7月  19 2018 mapred-site.xml.template
-rw-r--r--. 1 huser ftp    10 7月  19 2018 slaves
-rw-r--r--. 1 huser ftp  2316 7月  19 2018 ssl-client.xml.example
-rw-r--r--. 1 huser ftp  2697 7月  19 2018 ssl-server.xml.example
-rw-r--r--. 1 huser ftp  2250 7月  19 2018 yarn-env.cmd
-rw-r--r--. 1 huser ftp  4567 7月  19 2018 yarn-env.sh
-rw-r--r--. 1 huser ftp   690 7月  19 2018 yarn-site.xml
[root@master hadoop]#
```

### hadoop-env.sh

```shell
[root@master hadoop]# vim hadoop-env.sh

export JAVA_HOME=${JAVA_HOME}
修改成自己的 java 文件目录
export JAVA_HOME=/software/java
```


### core-site.xml



```xml
[root@master hadoop]# vim core-site.xml

<property>
<name>fs.defaultFS</name>
<value>hdfs://master:9000</value>
</property>
<property>
<name>hadoop.tmp.dir</name>
<value>/software/hadoop_tmp</value>
</property>
```
注意啊 
master 是自己的主机名字，不确定的话可以用 `hostname` 命令查一下

### yarn 配置

#### yarn-site.xml

```xml
[root@master hadoop]# vim yarn-site.xml
[root@master hadoop]#

<!-- NodeManager上运行的附属服务。需配置成mapreduce_shuffle，才可运行MapReduce程序 -->
<property>
<name>yarn.nodemanager.aux-services</name>
<value>mapreduce_shuffle</value>
</property>
```

#### mapred-site.xml

```shell
[root@master hadoop]#
[root@master hadoop]#
[root@master hadoop]#
[root@master hadoop]# cp mapred-site.xml.template mapred-site.xml
[root@master hadoop]#
[root@master hadoop]#
[root@master hadoop]#
[root@master hadoop]# vim mapred-site.xml
[root@master hadoop]#

<!-- 通知框架MR使用YARN -->
<property>
<name>mapreduce.framework.name</name>
<value>yarn</value>
</property>
```

## 丛机配置

```shell
[root@master hadoop]#
[root@master hadoop]# vim slaves
[root@master hadoop]#

slave1
slave2
```

## 发送文件给丛机

```shell

[root@master software]#
[root@master software]# pwd
/software
drwxr-xr-x. 9 huser ftp 149 7月  19 2018 hadoop
drwxr-xr-x. 7    10 143 245 7月   7 2018 java
[root@master software]#
[root@master software]# scp -r hadoop/ slave1:/software/
[root@master software]# scp -r hadoop/ slave2:/software/
[root@master software]# scp -r java/ slave1:/software/
[root@master software]# scp -r java/ slave2:/software/
[root@master software]# scp /etc/profile slave1:/etc/
[root@master software]# scp /etc/profile slave2:/etc/
```

刷新丛机的环境变量
```shell
. /etc/profile
```

## 格式化 NameNode

在 master 机器上执行
```shell
hadoop namenode -format
```
```shell

[root@master software]# ll
总用量 0
drwxr-xr-x. 9 huser ftp  149 7月  19 2018 hadoop
drwxr-xr-x. 3 root  root  17 5月   9 13:31 hadoop_tmp
drwxr-xr-x. 7    10  143 245 7月   7 2018 java
[root@master software]#
[root@master software]# cd hadoop_tmp/
[root@master hadoop_tmp]# ll
总用量 0
drwxr-xr-x. 3 root root 18 5月   9 13:31 dfs
[root@master hadoop_tmp]# cd dfs/
[root@master dfs]# ll
总用量 0
drwxr-xr-x. 3 root root 21 5月   9 13:31 name
```
在上面文件夹路径中有 name 文件夹说明格式化成功了

## 启动 HDFS

```shell
[root@master software]# cd hadoop
[root@master hadoop]# ll
总用量 112
drwxr-xr-x. 2 huser ftp   194 7月  19 2018 bin
drwxr-xr-x. 3 huser ftp    20 7月  19 2018 etc
drwxr-xr-x. 2 huser ftp   106 7月  19 2018 include
drwxr-xr-x. 3 huser ftp    20 7月  19 2018 lib
drwxr-xr-x. 2 huser ftp   239 7月  19 2018 libexec
-rw-r--r--. 1 huser ftp 86424 7月  19 2018 LICENSE.txt
-rw-r--r--. 1 huser ftp 14978 7月  19 2018 NOTICE.txt
-rw-r--r--. 1 huser ftp  1366 7月  19 2018 README.txt
drwxr-xr-x. 2 huser ftp  4096 7月  19 2018 sbin
drwxr-xr-x. 4 huser ftp    31 7月  19 2018 share
[root@master hadoop]# sbin
-bash: sbin: 未找到命令
[root@master hadoop]# cd sbin/
[root@master sbin]# ll
总用量 120
-rwxr-xr-x. 1 huser ftp 2752 7月  19 2018 distribute-exclude.sh
-rwxr-xr-x. 1 huser ftp 6452 7月  19 2018 hadoop-daemon.sh
-rwxr-xr-x. 1 huser ftp 1360 7月  19 2018 hadoop-daemons.sh
-rwxr-xr-x. 1 huser ftp 1640 7月  19 2018 hdfs-config.cmd
-rwxr-xr-x. 1 huser ftp 1427 7月  19 2018 hdfs-config.sh
-rwxr-xr-x. 1 huser ftp 2291 7月  19 2018 httpfs.sh
-rwxr-xr-x. 1 huser ftp 3128 7月  19 2018 kms.sh
-rwxr-xr-x. 1 huser ftp 4080 7月  19 2018 mr-jobhistory-daemon.sh
-rwxr-xr-x. 1 huser ftp 1648 7月  19 2018 refresh-namenodes.sh
-rwxr-xr-x. 1 huser ftp 2145 7月  19 2018 slaves.sh
-rwxr-xr-x. 1 huser ftp 1779 7月  19 2018 start-all.cmd
-rwxr-xr-x. 1 huser ftp 1471 7月  19 2018 start-all.sh
-rwxr-xr-x. 1 huser ftp 1128 7月  19 2018 start-balancer.sh
-rwxr-xr-x. 1 huser ftp 1401 7月  19 2018 start-dfs.cmd
-rwxr-xr-x. 1 huser ftp 3734 7月  19 2018 start-dfs.sh
-rwxr-xr-x. 1 huser ftp 1357 7月  19 2018 start-secure-dns.sh
-rwxr-xr-x. 1 huser ftp 1571 7月  19 2018 start-yarn.cmd
-rwxr-xr-x. 1 huser ftp 1347 7月  19 2018 start-yarn.sh
-rwxr-xr-x. 1 huser ftp 1770 7月  19 2018 stop-all.cmd
-rwxr-xr-x. 1 huser ftp 1462 7月  19 2018 stop-all.sh
-rwxr-xr-x. 1 huser ftp 1179 7月  19 2018 stop-balancer.sh
-rwxr-xr-x. 1 huser ftp 1455 7月  19 2018 stop-dfs.cmd
-rwxr-xr-x. 1 huser ftp 3206 7月  19 2018 stop-dfs.sh
-rwxr-xr-x. 1 huser ftp 1340 7月  19 2018 stop-secure-dns.sh
-rwxr-xr-x. 1 huser ftp 1642 7月  19 2018 stop-yarn.cmd
-rwxr-xr-x. 1 huser ftp 1340 7月  19 2018 stop-yarn.sh
-rwxr-xr-x. 1 huser ftp 4295 7月  19 2018 yarn-daemon.sh
-rwxr-xr-x. 1 huser ftp 1353 7月  19 2018 yarn-daemons.sh
[root@master sbin]# cd ..
[root@master hadoop]# ll
总用量 112
drwxr-xr-x. 2 huser ftp   194 7月  19 2018 bin
drwxr-xr-x. 3 huser ftp    20 7月  19 2018 etc
drwxr-xr-x. 2 huser ftp   106 7月  19 2018 include
drwxr-xr-x. 3 huser ftp    20 7月  19 2018 lib
drwxr-xr-x. 2 huser ftp   239 7月  19 2018 libexec
-rw-r--r--. 1 huser ftp 86424 7月  19 2018 LICENSE.txt
-rw-r--r--. 1 huser ftp 14978 7月  19 2018 NOTICE.txt
-rw-r--r--. 1 huser ftp  1366 7月  19 2018 README.txt
drwxr-xr-x. 2 huser ftp  4096 7月  19 2018 sbin
drwxr-xr-x. 4 huser ftp    31 7月  19 2018 share

[root@master hadoop]# sbin/start-dfs.sh
Starting namenodes on [master]
The authenticity of host 'master (192.168.64.80)' can't be established.
ECDSA key fingerprint is SHA256:64n0mG4pDvFJvVUUyCHb5zNc1OqMflsj+MLvtqktW74.
ECDSA key fingerprint is MD5:fc:23:a0:42:5d:82:17:4c:93:b6:7d:cc:c2:3b:48:c7.
Are you sure you want to continue connecting (yes/no)? yes
master: Warning: Permanently added 'master,192.168.64.80' (ECDSA) to the list of known hosts.
master: starting namenode, logging to /software/hadoop/logs/hadoop-root-namenode-master.out
slave2: starting datanode, logging to /software/hadoop/logs/hadoop-root-datanode-slave2.out
slave1: starting datanode, logging to /software/hadoop/logs/hadoop-root-datanode-slave1.out
Starting secondary namenodes [0.0.0.0]
The authenticity of host '0.0.0.0 (0.0.0.0)' can't be established.
ECDSA key fingerprint is SHA256:64n0mG4pDvFJvVUUyCHb5zNc1OqMflsj+MLvtqktW74.
ECDSA key fingerprint is MD5:fc:23:a0:42:5d:82:17:4c:93:b6:7d:cc:c2:3b:48:c7.
Are you sure you want to continue connecting (yes/no)? yes
0.0.0.0: Warning: Permanently added '0.0.0.0' (ECDSA) to the list of known hosts.
0.0.0.0: starting secondarynamenode, logging to /software/hadoop/logs/hadoop-root-secondarynamenode-master.out
```

## 启动 yarn

```shell
[root@master hadoop]# sbin/start-yarn.sh
starting yarn daemons
starting resourcemanager, logging to /software/hadoop/logs/yarn-root-resourcemanager-master.out
slave2: starting nodemanager, logging to /software/hadoop/logs/yarn-root-nodemanager-slave2.out
slave1: starting nodemanager, logging to /software/hadoop/logs/yarn-root-nodemanager-slave1.out
[root@master hadoop]#
```

启动成功之后可以看到

在 master 机器上
```shell
[root@master hadoop]# pwd
/software/hadoop
[root@master hadoop]#
[root@master hadoop]# jps
2545 ResourceManager
2808 Jps
2217 NameNode
2399 SecondaryNameNode
```
在 slave 机器上看到启动的进程，还有 data 文件
```shell
[root@slave1 software]# jps
9508 DataNode
9604 NodeManager
9702 Jps
[root@slave1 software]#
[root@slave1 dfs]# ll /software/hadoop_tmp/dfs/
总用量 0
drwx------. 3 root root 40 5月   9 13:39 data
[root@slave1 dfs]#
```

## web GUI

http://192.168.64.80:50070/

---

参考：
https://blog.csdn.net/weixin_44964753/article/details/102635174
https://www.bilibili.com/video/av71702657/
https://www.bilibili.com/video/BV1FJ411T7MU/?spm_id_from=333.788.b_7265636f5f6c697374.1
https://www.bilibili.com/video/BV1C5411c7Qv?p=3&spm_id_from=pageDriver
https://www.bilibili.com/video/BV1Xt4y1q7Rj/?spm_id_from=333.788.recommend_more_video.0