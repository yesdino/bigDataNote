# 目录

[toc]

---



[link](https://www.bilibili.com/video/BV1W4411B7cN?p=1)


---

# 第1章 Hive基本概念

[06:10](https://www.bilibili.com/video/BV1W4411B7cN?p=2)


**Hive**: 由 Facebook 开源用于解决 海量 **<u>结构化日志</u>** 的数据统计。
- 基于 Hadoop 的一个 数据仓库(数仓) **工具**，
- ==可以将结构化的 **数据文件** 映射为 **一张表**==，并提供 <u>类 SQL</u> 查询功能
- **本质是将 HQL 转化成 Mapreduce 程序**


hive sql 执行流程：
<img style="width:800px" src="https://upload-images.jianshu.io/upload_images/11876740-51a60e3e7a6656f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></img>
1）Hive 处理的数据存储在 HDFS
2）Hive 分析数据底层的默认实现是 MapReduce
3）执行程序运行在 Yarn 上

所以这样看来：==hive 相当于 hadoop 的客户端== 



## 1.2 Hive 的优缺点

### 1.2.1 优点

- 1）避兔了去写 Mapreduce, 减少开发人员的学习成本
- 2）操作接口采用类 SQL 语法，提供快速开发的能力 ( 简单、容易上手 )
- 3）Hive 的执行 **延迟比较高**，因此 Hive 常用于数据分析，**对实时性要求不高** 的场合
- 4）Hive **优势在于处理大数据**，**对于处理小数据没有优势**，因为 Hive 的执行延迟比较高
- 5）Hive 支持用户自定义函数，用户可以根据自己的需求来实现自己的函数

<br>

### 1.2.2 缺点

[00:00](https://www.bilibili.com/video/BV1W4411B7cN?p=4)
- 1）Hive 的 HQL 表达能力有限
    - (1) 迭代式算法无法表达。（迭代式算法即需要对某些数据反反复复进行计算）
    - (2) 数据挖掘方面不擅长
<br>

- 2）Hive 的效率比较低
    - (1) Hive 自动生成的 Mapreduce作业，通常情况下不够智能化
    - (2) Hive **==调优比较困难==**，粒度较粗。(因为 hql 是由 MapReduce 转化而来的模板，模板固定了是比较难调优的)
  
<br>


## 1.3 Hive 架构原理

[00:18](https://www.bilibili.com/video/BV1W4411B7cN?p=4)


<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/11876740-045a73ea1fd6284c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></img>





**1）用户接口： Client**
CLl(hive shell) 、JDBC/ODBC(ava 访问 hive) 、 WEBUI (浏览器访问 hive)

**2）元数据： Meta store**
元数据包括：表名、表所属的数据库 ( 默认是 default) 、表的拥有者、列/分区字段、表
的类型（是否是外部表）、表的数据所在目录等；
==默认存储在自带的 derby 数据库中，推荐使用 MySQL 存储 Metastore==

**3）Hadoop.**
使用 HDFS 进行存储，使用 Mapreduce 进行计算

**4）驱动器： Driver.**
(1) 解析器 ( SQL Parser): 将 SQL 字符串转换成抽象语法树 AST, 这一步一般都用第三方工具库完成，比如 antlr; 对 AST 进行语法分析，比如表是否存在、字段是否存在、 SQL 语义是否有误
(2) 编译器 ( Physical Plan): 将 AST 编译生成逻辑执行计划。
(3) 优化器 ( Query Optimizer): 对逻辑执行计划进行优化。
(4) 执行器 ( Execution): 把逻辑执行计划转换成可以运行的物理计划。对于 Hive 来说，就是 MR/Spark 。

<br><br><br><br><br><br>
<br>

---

# 第2章 Hive安装








<br><br><br><br><br><br><br>

---

# 第3章 Hive数据类型






<br><br><br><br><br><br><br>

---

# 第4章 DDL数据定义






<br><br><br><br><br><br><br>

---

# 第5章 DML数据操作*





<br><br><br><br><br><br><br>

---

# 第6章 查询*






<br><br><br><br><br><br><br>

---

# 第7章 函数*

## 7.1系统内置函数





<br><br><br><br><br><br><br>


## 7.2 自定义函数





<br><br><br><br><br><br><br>

### 7.3.1 自定义UDF函数




<br><br><br><br><br><br><br>

### 7.3.2 自定义UDTF函数





<br><br><br><br><br><br><br>


# 第8章 压缩和存储


# 第9章 企业级调优


# 第10章 Hive实战之谷粒影音


# 第11章 常见错误及解决方案