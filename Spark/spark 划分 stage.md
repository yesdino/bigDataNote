# 目录

[toc]

---



## 1、为什么要划分 stage（调度阶段）？
[link](https://www.bilibili.com/video/BV1m5411p7wR?from=search&seid=14248942010279108564&t=16m30s)
由于个job任务中可能有大量的宽窄依赖，由于窄依赖不会产生 shuffle,宽依赖会产生 shuffle
后期划分完 stage 之后，在同一个 stager中只有窄依赖，并没有宽依赖，这些窄依赖对应的 task 是可以互相独立的去运行
划分完 stage 之后，它内部是有很多可以并行运行的 task


## 2、如何划分 'stage?

[link](https://www.bilibili.com/video/BV1m5411p7wR?from=search&seid=14248942010279108564&t=17m12s)

划分 stage 就是以宽依进行划分

(1)、生成DAG有向无环图之后，从最后个rdd往前推，先创建一个 stage,它是最后一个 stage
(2)、如果遇到了窄依赖就把该rdd加入到 stager中，如果遇到了宽依赖，就从宽依赖切开，最后一个 ' staget也就划分结束了
(3)、后面重新创建一个新的 stage,还是按照第二步操作继续往前推，一直推到最开始的rdd,整个划分 stage也就结束了




## 3、stagel内部的逻辑

[link](https://www.bilibili.com/video/BV1m5411p7wR?from=search&seid=14248942010279108564&t=27m37s)

(1)每一个 stage中按照rdd的分区划分成了很多个可以并行运行的task
(2)把每个 'stager中这些可以并行运行的task都封装在一个 .task Set集合中
(3)rd与rd之间存在对应的依赖关系， stage与 stage.之间也存在对应的依赖关系，前面 Stage的task先运
行，运行完成之后，后面 stager中的task再运行，也就是说前面 stager中task它的输出结果数据，是后面 stage
中task输入数据。


## 4、Application、Job、Stage、Task 之间的关系

[link](https://www.bilibili.com/video/BV1m5411p7wR?from=search&seid=14248942010279108564&t=30m54s)

application 是 spark 的一个应用程序，它是包含了客户端写好的代码以及任务运行的时候需要的资源信息。
后期再一个 application 中有很多个 action 操作，一个 action 操作就是一个 job, 1个 job 会存在大量的宽依赖，后期
按照宽依赖进行 stage 的划分，个 job 又产生了很多个 stage, 每一个 stage 内部有很多并行运行的 task.

<img style="width:700px" src="img\stage_job划分.png"></img>
