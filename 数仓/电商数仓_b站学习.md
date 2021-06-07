# 目录

[toc]

---

[link](https://www.bilibili.com/video/BV1df4y1U79z?from=search&seid=15154304885674956173)



# 第1章 数据仓库概念

**业务数据**：

就是各行业 **在处理事务过程中产生的数据**。
比如 <u>用户在电商网站中登录、下单、支付等过程中产生的数据</u>就是业务数据。
业务数据通常存储在 MYSQL、 Oracle等数据库中。
[link](https://www.bilibili.com/video/BV1df4y1U79z?p=2)


**用户行为数据**：

用户在使用产品过程中，**与客户端产品交互过程中产生的数据**，
比如页面浏览、点击、停留、评论、点赞、收藏等。<u></u>
用户行为数据通常存储在日志文件中。
（**反应用户心理变化的数据**，比如对某一商品的喜爱程度，得到这些数据可以后续做相关的推荐）
[00:39 看 url 中的参数解析，淘宝是如何抓取用户行为数据的](https://www.bilibili.com/video/BV1df4y1U79z?p=2)

<img width=900 src='https://upload-images.jianshu.io/upload_images/11876740-a4b33722b6472a1e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

上面抓取的是 **半结构化数据**，存储在日志文件中，
后面会将半结构数据转换成结构化数据，放在 MySQL 中存储或放在 hive 数仓中进行正常解析 

网站为了获取与用户行为相关的半结构化数据，可以在网站中 **埋点**                                                                                                                     


## 数据仓库
数据仓库(Data Warehouse)，是**为企业制定决策，提供数据支持**的。
可以帮助企业，改进业务流程、提高产品质量等。

[00:00](https://www.bilibili.com/video/BV1df4y1U79z?p=4)
<img width=920 src='https://upload-images.jianshu.io/upload_images/11876740-9edb63d95fddde6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240'>

Azkaban 全流程调度：自动调度数据处理的流程的框架，比如 ODS 层处理完了会自动进行 DWD 层的处理

### ① 数据来源

[01:00](https://www.bilibili.com/video/BV1df4y1U79z?p=4)
- 获取数仓数据来源的方式主要是爬虫
- 获取的数据主要有两种：用户行为数据、业务数据
    - **用户行为数据** 通常以文件形式存储在日志服务器
    - **业务数据** 通常以数据表形式存储在 javae 后台的 MySQL 数据库中  


### ② 中间处理


**`Flume`**: 用来处理用户行为数据，导入用户行为数据的日志文件，可以实时采集日志文件，上传到数据仓库中
**`Sqoop`**: 每天凌晨结束后，采集==前一天==的数据，导入数据仓库中


当不同的数据通过  和  进入数据仓库中，会经过 `ODS -> DWD -> DWS -> DWT -> ADS` 的流程

**`ODS`** ：将数据先备份。为了避免后续数据处理出现问题导致需要重复采集数据
**`DWD`** ：清洗数据。处理过期数据，不完整数据，重复数据等脏数据
**`DWS`** ：聚合数据。按天进行聚合，既用户在一天中做了什么操作
**`DWT`** ：聚合数据。累积聚合，既用户从注册开始到现在累积做了什么
**`ADS`** ：统计数据。统计分析完的数据，这层数据是数据的统计，是输入之前的数据形式。


数据仓库，并不是数据的最终目的地，而是为数据最终的目的地做好准备。
这些准备包括对数据的：备份、清洗、聚合、统计等。


### ③ 数据输出

[06:45](https://www.bilibili.com/video/BV1df4y1U79z?p=4&t=6m45s)

数据仓库 ADS 层的数据将会输出到下面的出处：
- **报表系统**：统计好的数据输出到报表中给领导汇报
- **用户画像**：描绘用户画像
- **推荐系统**：
- **机器学习**：为机器学习提供训练数据



## 项目需求

[link](https://www.bilibili.com/video/BV1df4y1U79z?p=5)

<img width=100% src="../img/数仓/项目需求.png"></img>



## 技术选型

[link](https://www.bilibili.com/video/BV1df4y1U79z?p=6)

<img width=100% src="../img/数仓/技术选型.png"></img>

[link 为什么要选这些框架](https://www.bilibili.com/video/BV1df4y1U79z?p=7)

<img width=70% src="../img/数仓/技术选型2.png"></img>



## 系统架构

[link](https://www.bilibili.com/video/BV1df4y1U79z?p=8&spm_id_from=pageDriver)

<img width=100% src="../img/数仓/系统架构.png"></img>

**业务交互数据**：
<b style="color:red"> 业务流程中产生的登录、订单、用户商品、支付等相关的数据</b>，通常存储在DB中，包括 Mysql、 Oracle 等

**埋点用户行为数据**：
<b style="color:red"> 用户在使用产品过程中，与客户端产品交互过程中产生的数据</b>，比如页面测览、点击、停留、评论、点赞收藏等






<br><br><br><br><br><br><br><br><br>

---


<img width=0% src=""></img>

<b style="color:red"></b>



















<u></u>

<br>
<br>
<br>
<br>
<br>
<br>









