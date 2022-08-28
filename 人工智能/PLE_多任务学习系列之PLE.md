# 目录

[toc]

---

[source](https://zhuanlan.zhihu.com/p/408631841)

# 多任务学习系列之PLE


多任务学习的典型工作有多独立塔 DNN ，多头 DNN ， MOE ， MMOE 等工作，
今天介绍的是腾讯的 **`PLE (Progressive Layered Extraction)`** 模型， 

==PLE 重点解决了多任务学习中存在的跷跷板现象（seesaw phenomenon）。==

多任务学习（MTL）并被证明 ==**可以通过任务之间的信息共享来提高学习效率**==。 

**多任务学习缺点**：
1. <blue>负迁移</blue>
然而，多个任务经常是松散相关甚至是相互冲突的，这可能导致性能恶化，这种情况称为 **==负迁移==** 。 
2. <blue>跷跷板现象</blue>
在论文中提到，通过在真实世界的大规模视频推荐系统和公共基准数据集上的大量实验，
发现现有的 MTL 模型经常以牺牲其他任务的性能为代价来改进某些任务，
当任务相关性很复杂并且有时依赖于样本时，
即与相应的单任务模型相比，多个任务无法同时改进，论文中称之为 ==**跷跷板现象**==。

为了解决跷跷板和负迁移的现象，
论文提出了一种 **共享结构设计的渐进式分层提取**（ PLE ）模型。
其包含两部分， 
- 一部分是一种 **==显式区分共享专家塔和特定任务专家塔==的 门控 (CGC) 模型**，
- 另一部分是 **==由单层 CGC 结构扩展到多层== 的 PLE 模型**。


## 1 CGC ( Customized Gate Control )
为了实现和单任务相似的性能，论文显式分离了任务共享专家部分和任务独享专家部分，并提出了 CGC （如下图所示）。
顶部是一些和任务相关的多层塔网络，底部是一些专家模块，每个专家模块由多个专家网络组成。
专家模块分为两类，
- 一类是任务共享的专家模块，负责学习任务的共享模式，
- 一类是任务独享的专家模块，负责学习任务的独享模式。

每个塔网络从所有专家模块（共享专家模块和独享专家模块）学习知识。

<img width="550" src="https://pic4.zhimg.com/80/v2-4bb86790bab26a1e95f445ab4009651b_720w.jpg"></img>



在 CGC 中，所有专家模块的输出通过一个门控网络融合。
门控网络是一个单层前馈网络，使用 softmax 作为激活函数。 
input 作为选择器计算所有选择向量的加权和（input as the selector to calculate the weighted sum of the selected vectors），
这句话有点绕，
其实就是 **基于 input 生成选择概率，然后基于选择概率融合所有专家模块的输出**，
所以 input 被称作 selector ；
 ，它们基于选择概率被选择了。
任务 k 的门控网络形式化如下：
<img width="550" src="https://pic1.zhimg.com/80/v2-83771b5c32e5fa4306d9de2f98c514d0_720w.jpg"></img>

上面的公式其实很简单，$W^k(x)$  就是门控网络的参数矩阵， $S^k(x)$ ​是所有专家模块的输出，  $g^k(x)$ 是门控网络的输出，任务k的输出就是 $y^k(x)$ ，$t^k(x)$  是任务k的塔网络。


---

## 2 PLE ( Progressive Layered Extraction )

PLE 将单层 CGC 结构扩展到了多层，用于学习越来越深的语义表示。
那么在多层 CGC 中，**独享专家模块和共享专家模块怎么融合上一层网络的输出** 是一个需要考虑的问题。
为此，论文提出了多级提取网络，其结构类似于 CGC，但是定义了各个专家模块更新的逻辑。
多级提取网络的结构如下图所示。

<img width="600" src="https://pic4.zhimg.com/80/v2-c4a84175193eced0c5a5d41761d5434f_720w.jpg"></img>

在多级提取网络中，
任务 k 的独享专家模块融合了上一层网络中任务 k 的独享专家模块和共享专家模块，
而共享专家模块则融合了上一层所有的专家模块。
所以在多级提取网络中有两个门控， for 独享专家模块和 for 共享专家模块。
所以在 PLE 中，不同任务的参数并不是像 CGC 中完全分离的，而是在顶层网络中分离的。
在高层提取网络中的门控网络将低层提取网络中门控的融合结果作为选择器，而不是 input 。
论文给出的理由是 在高层 expert 网络中可以为选择提供更好的抽象信息 
(it may provide better information for selecting abstract knowledge extracted in higher-level experts) 。

PLE 中权重函数、选择矩阵和门控网络的计算与 CGC 中的相同。 
具体来说， PLE 的第 j 个提取网络中任务 k 的门控网络的公式为：
<img width="600 " src="https://pic2.zhimg.com/80/v2-b9b1d67bc1766349a21078ecf4f70065_720w.jpg"></img>

和 MMOE 相比， PLE ==增加了不同的 Expert 之间交互。==


---

## 3 loss 设计

loss 好像没有什么特别的地方。


$\delta^i_k$ 表示样本 i 是否在任务 k 的样本空间中  
(indicates whether sample i lies in the sample space of task k)

<img width="600" src="https://pic4.zhimg.com/80/v2-d0fcf60e35e3f62c684c996d0ff2b77f_720w.jpg"></img>

参考文献：

[1] [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/pdf/10.1145/3383313.3412236)

[2] [绝密伏击：多目标学习在推荐系统的应用(MMOE/ESMM/PLE)](https://zhuanlan.zhihu.com/p/291406172)




<br>
<br><br><br><br><br><br>


<u></u>

<!-- 
<img width="" src=""></img>
<img style="width:500px" src=""></img>
 -->


<style>
.red {
	color: red;
	font-weight: bold;
}
blue {
	color: blue;
	font-weight: bold;
}


</style>



