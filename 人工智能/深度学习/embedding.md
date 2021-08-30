# 目录

[toc]

---

[source](http://jalammar.github.io/illustrated-word2vec/)
[source](https://www.jianshu.com/p/b110c06954a6)

# 图解word2vec

<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-7620eb198cc6b82f?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

词嵌入（embedding）是机器学习中最惊人的创造， 如果你有使用过 Siri、Google Assistant、Alexa、Google 翻译，输入法打字预测等经历，那么你很有可能是词嵌入（自然语言处理的核心模型）技术的受益者。

在过去的几十年中，基于神经网络模型的词嵌入技术发展迅猛。
尤其是最近，包括使用 BERT 和 GPT2 等最先进的模型进行语义化词嵌入。

Word2vec 是一种有效创建词嵌入的方法，于 2013 年提出。
除了作为词嵌入的方法之外，它的一些概念已经被证明可以有效地应用在推荐引擎和理解时序数据上。
像 Airbnb 、阿里巴巴、 Spotify 这样的公司都有使用 Word2vec 并用于产品中，为推荐引擎提供支持。

在本文中，将讨论词嵌入的概念，以及使用 word2vec 生成词嵌入的机制。
从一个例子开始，熟悉使用向量来表示事物，你是否知道你的个性可以仅被五个数字的列表（向量）表示？

# 1. 词嵌入表示你的性格

如何用 0 到 100 的范围来表示你是多么内向 / 外向（其中 0 是最内向的， 100 是最外向的）？ 
你有没有做过像 MBTI 那样的人格测试，或者五大人格特质测试？ 
如果你还没有，这些测试会问你一系列的问题，然后在很多维度给你打分，**内向/外向** 就是其中之一。
<img style="width:400px" src="https://upload-images.jianshu.io/upload_images/19380915-f360c532f9257280?imageMogr2/auto-orient/strip|imageView2/2/w/439/format/webp"></img>
五大人格特质测试测试结果示例。
它可以真正告诉你很多关于你自己的事情，并且在学术、人格和职业成功方面都具有预测能力。

假设我的内外向得分为 **38/100**。 可以用这种方式绘图表示：

<img style="width:400px" src="https://upload-images.jianshu.io/upload_images/19380915-22e5ae96f9e3a164?imageMogr2/auto-orient/strip|imageView2/2/w/393/format/webp"></img>

把范围收缩到 -1 到 1:

<img style="width:400px" src="https://upload-images.jianshu.io/upload_images/19380915-217d3475a1a931e7?imageMogr2/auto-orient/strip|imageView2/2/w/401/format/webp"></img>

当你只知道这一条信息的时候，你觉得你有多了解这个人？
人性很复杂，需要添加另一测试的得分作为新维度。

<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-75c701be0f4f2b90?imageMogr2/auto-orient/strip|imageView2/2/w/473/format/webp"></img>

可以将两个维度表示为图形上的一个点，或者作为 **从原点到该点的向量**。

现在可以说这个向量代表了我的一部分人格。
当你想要将另外两个人与我进行比较时，这种表示法就有用了。
假设我突然被车撞挂了，需要找一个性格相似的人替换我，
那在下图中，两个人中哪一个更像我？
<img style="width:550px" src="https://upload-images.jianshu.io/upload_images/19380915-3b1cdfab0c12ed37?imageMogr2/auto-orient/strip|imageView2/2/w/439/format/webp"></img>
处理向量时，计算相似度得分的常用方法是使用 ==**余弦相似度**==：
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-a6cd68cafca8f4ea?imageMogr2/auto-orient/strip|imageView2/2/w/494/format/webp"></img>
很明显，1号替身在性格上与我更相似。
指向相同方向的向量（长度也起作用）具有更高的余弦相似度。

两个维度还不足以捕获有关不同人群的足够信息。
心理学已经研究出了五个主要人格特征（以及大量的子特征），所以使用所有五个维度进行比较：
<img style="width:350px" src="https://upload-images.jianshu.io/upload_images/19380915-95ed1cfe8631dc3e?imageMogr2/auto-orient/strip|imageView2/2/w/300/format/webp"></img>

使用五个维度的问题是不能在二维平面绘制整齐小箭头了。
在机器学习中，经常需要在更高维度的空间中思考问题。 
但是 ==**余弦相似度的计算仍然有效，仍适用于任意维度**==：

<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-a806e66bc57d991b?imageMogr2/auto-orient/strip|imageView2/2/w/554/format/webp"></img>

余弦相似度适用于任意维度的数据，现在的得分比上次的得分要更合理，因为是根据被比较事物的更高维度算出的。

在本节的最后，总结两个中心思想：
- 1- **==可以将人和事物表示为代数向量（这样才能使用机器进行！）。==**
- 2- **==可以很容易地计算出向量之间的相互关系，是否相似。==**
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-6659dbc882ffc141?imageMogr2/auto-orient/strip|imageView2/2/w/548/format/webp"></img>

---

# 2. Word Embeddings（词嵌入）

通过上文的理解，继续看看 **训练好的词向量（也被称为词嵌入）**，并接着探索它们的一些有趣属性。

这是一个单词 "**`king`**" 的词嵌入（在维基百科上训练的GloVe向量）：

```shell
[ 0.50451 , 0.68607 , -0.59517 , -0.022801, 0.60046 , -0.13498 , -0.08813 , 0.47377 , -0.61798 , -0.31012 , -0.076666, 1.493 , -0.034189, -0.98173 , 0.68229 , 0.81722 , -0.51874 , -0.31503 , -0.55809 , 0.66421 , 0.1961 , -0.13495 , -0.11476 , -0.30344 , 0.41177 , -2.223 , -1.0756 , -1.0783 , -0.34354 , 0.33505 , 1.9927 , -0.04234 , -0.64319 , 0.71125 , 0.49159 , 0.16754 , 0.34344 , -0.25663 , -0.8523 , 0.1661 , 0.40102 , 1.1685 , -1.0137 , -0.21585 , -0.15155 , 0.78321 , -0.91241 , -1.6106 , -0.64426 , -0.51042 ]
```
这是一个包含 50 个数字的列表。
通过观察数值看不出什么，但是稍微给它可视化，就可以和其它词向量作比较。
把上面所有这些数字放在一行：

<img width=900px src="https://upload-images.jianshu.io/upload_images/19380915-2d960ecf731bfa48?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

根据它们的值对单元格进行颜色编码（如果它们接近 2 则为红色，接近 0 则为白色，接近 -2 则为蓝色），得到如下图片：
<img width=1000px src="https://upload-images.jianshu.io/upload_images/19380915-589746e404613b53?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
忽略数字仅查看单元格颜色。现在将"king"与其它单词进行比较：
<img width=550px src="https://upload-images.jianshu.io/upload_images/19380915-5097dde8614fa046?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

上图可以看出："Man"和"Woman"彼此之间要更加相似，相比于单词"king"来说。 
这暗示着向量图示很好的展现了这些单词的信息/含义/关联。

这是另一个示例列表：
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-ed8b57b3a098eefe?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

总结：

- 1、所有这些不同的单词都有一条直的红色列，说明它们在这个维度上是相似的（虽然不知道每个维度具体是什么含义）
- 2、可以看到"woman"和"girl"在很多维度是相似的，"man"和"boy"也是一样
- 3、"boy"和"girl"也有彼此相似的地方，但这些地方却与"woman"或"man"不同。这些维度是否可以总结出一个模糊的"youth"概念？
- 4、除了最后一个单词，所有单词都是代表人。 通过添加了一个"water"对象来显示类别之间的差异。你可以看到蓝色列一直向下并在 "water"的词嵌入之前停下了。
- 5、"king"和"queen"彼此之间相似，但它们与其它单词都不太相似。这些不相似的维度是否可以总结出一个模糊的"royalty"概念？

## 2.1 类比

展现词嵌入奇妙属性的著名例子是类比。
通过添加、减去词嵌入得到一些有趣的结果。一个经典例子是公式：**`"king"-"man"+"woman"`**：
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-610138a2a9bfed58?imageMogr2/auto-orient/strip|imageView2/2/w/535/format/webp"></img>

在python中使用 Gensim 库，可以添加和减去词向量，它会找到与结果向量最相似的单词。
该图像显示了最相似的单词列表，输出其与每个单词余弦相似性。

同样进行可视化：
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-d78e93cc1d71f83a?imageMogr2/auto-orient/strip|imageView2/2/w/558/format/webp"></img>

虽然 **`"king-man + woman"`** 生成的向量并不完全等同于 **`"queen"`**，
但是 **`"queen"`** 是在整个 400,000 个字词嵌入中最接近它的单词。

由此已经看过训练好的词嵌入向量了，下文会详细介绍训练过程。 
在开始使用 word2vec 之前，需要看一下词嵌入的上级概念：**神经语言模型**。

---

#  3. Language Modeling 语言模型

自然语言处理最典型的应用，应该就是智能手机输入法中对下一单词预测功能。
<img style="width:300px" src="https://upload-images.jianshu.io/upload_images/19380915-7d678e1faaafcca9?imageMogr2/auto-orient/strip|imageView2/2/w/387/format/webp"></img>

单词预测可以通过语言模型来实现，语言模型会**从单词列表中(比如说两个词)去尝试预测可能紧随其后的单词**。

在上面这个手机截屏中，可以认为该模型接收到两个绿色单词(thou shalt)并推荐了一组单词("not" 就是其中最有可能被选用的一个)：
<img style="width:500px" src="https://upload-images.jianshu.io/upload_images/19380915-ec70c63ade11f7d6?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

可以把这个模型想象为如下所示的黑盒:
<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-f84a06550e658624?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

但事实上，该模型不会只输出一个单词。
实际上，所有它知道的单词（模型的词库，可能有几千到几百万个单词）**按可能性打分**，
输入法程序会 **选出其中分数最高的推荐给用户**。
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-6b1a57e507750028?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

自然语言模型的输出就是模型所知单词的概率评分，通常把概率按百分比表示，
实际使用中，40% 这样的分数在输出向量组是表示为 0.4

自然语言模型 (请参考Bengio 2003) 在完成训练后，会按下图所示的三个步骤完成预测：
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-559edb180e65a662?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

第一步与本文主题最相关，因为讨论的就是Embedding。
**模型在经过训练之后会生成一个 ==映射单词表所有单词的词向量矩阵==**。

在进行预测的时候，输入法预测算法就是在这个映射矩阵中查询输入的单词的词向量，
然后计算出预测值:
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-986960512611c1c5?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
下面将重点放到模型训练上，来学习一下模型如何构建这个映射矩阵词向量。

## 3.1 Language Model Training 语言模型训练

相较于大多数其他机器学习模型来说，语言模型有一个很大有优势，那就是有丰富的训练文本。
书籍、文章、维基百科、以及各种类型的文本内容都可用。
<u>相比之下，训练其他机器学习模型就需要手工标注数据或者专门采集数据。</u>

通过找出每个单词附近的词，通过就能获得它们的映射关系，步骤如下：
- ①、获取大量文本数据(例如维基百科内容)
- ②、然后建立一个可以沿文本滑动的窗(例如一个窗里包含三个单词)
- ③、利用这样的滑动窗就能为训练模型生成大量样本数据。

<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-f9f98b20c26bbc11?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

当这个窗口沿着文本滑动时，就可以生成用于模型训练的数据集。
为了明确理解这个过程，看下滑动窗是如何处理这个短语的:
> “Thou shalt not make a machine in the likeness of a human mind” ~Dune

在一开始的时候，窗口锁定在句子的前三个单词上:
<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-ee0513ad2d86514a?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
**==把前两个单词单做特征，第三个单词做标签==**:
<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-624d0dc89d7ea966?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
**这时就生产了数据集中的第一个样本**，会被用在后续的语言模型训练中。
接着，将窗口滑动到下一个位置并生产第二个样本:
<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-61ab6c38df4dc702?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
**这时第二个样本也生成了**。
重复上述步骤，就能得到一个较多样本的数据集，从数据集中能看到在不同的单词组后面会出现的单词:
<img style="width:600px" src="https://upload-images.jianshu.io/upload_images/19380915-881311c0b1467c2c?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
在实际应用中，模型往往在滑动窗口时就被训练的，而不是事先产生一个个的样本。

为了让读者更清晰易懂，本文将 **生成数据集** 和 **训练模型** 分为两个阶段来解释。
除了使用神经网络建模训练词向量，还常用 **N-gams** 的方法进行模型训练。


## 3.2 Look both ways 顾及两头
一是根据前面的信息进行填空:
<img style="width:220px" src="https://upload-images.jianshu.io/upload_images/19380915-f1e7a9a59b1145f3?imageMogr2/auto-orient/strip|imageView2/2/w/349/format/webp"></img>
在空白前面，提供的语义背景是五个单词(如果事先提及到`'bus'`)，可以肯定，大多数人都会把 bus 填入空白中。
但是如果我再给你一条信息------比如空白后的一个单词，那答案会有变吗？
<img style="width:250px" src="https://upload-images.jianshu.io/upload_images/19380915-1d1fc42752bcf8a1?imageMogr2/auto-orient/strip|imageView2/2/w/406/format/webp"></img>
这下空白处改填的内容完全变了。这时'red'这个词最有可能适合这个位置。
从这个例子中能学到，**一个单词的前后词语都带信息价值**。
事实证明，需要考虑两个方向的单词 (目标单词的左侧单词与右侧单词)。

那该如何调整训练方式以满足这个要求呢，继续往下看。

### 3.2.1 Skipgram 模型（生成样本数据）

**①、连续词袋(CBOW)**

不仅要考虑目标单词的前两个单词，还要考虑其后两个单词。

<img style="width:350px" src="https://upload-images.jianshu.io/upload_images/19380915-a4a0d22410e2fc72?imageMogr2/auto-orient/strip|imageView2/2/w/476/format/webp"></img>
如果这么做，实际上构建并训练的模型就如下所示：

<img style="width:400px" src="https://upload-images.jianshu.io/upload_images/19380915-4ba832009f84e3f5?imageMogr2/auto-orient/strip|imageView2/2/w/300/format/webp"></img>
上述的这种方法被称为 **Continuous Bag of Words 连续词袋(CBOW)**。

**②、Skipgram**

还有另一种想法，它不根据前后文(前后单词)来猜测目标单词，而是推测当前单词可能的前后单词，
设想一下滑动窗在训练数据时如下图所示：
<img style="width:450px" src="https://upload-images.jianshu.io/upload_images/19380915-a18531682a28392d?imageMogr2/auto-orient/strip|imageView2/2/w/467/format/webp"></img>
绿框中的词语是输入词，粉框则是可能的输出结果。
这里粉框颜色深度呈现不同，是因为滑动窗给训练集产生了4个独立的样本:

<img style="width:450px" src="https://upload-images.jianshu.io/upload_images/19380915-756a6808296e18ef?imageMogr2/auto-orient/strip|imageView2/2/w/441/format/webp"></img>
这种方式称为 **Skipgram 架构**。
可以像下图这样将展示滑动窗的内容。

<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-0e3b509adbd70f3a?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
这样就为数据集提供了4个样本:
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-620766b023bad8f3?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
然后移动滑动窗到下一个位置:
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-593b7ac2f91b78bc?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
这样又产生了接下来4个样本:
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-7be1db6f0e570812?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>
在移动几组位置之后，就能得到一批样本:
<img style="width:700px" src="https://upload-images.jianshu.io/upload_images/19380915-66f3ae394a9ea625?imageMogr2/auto-orient/strip|imageView2/2/w/1080/format/webp"></img>

### 3.2.2 训练过程
现在已经从现有的文本中获得了 Skipgram 模型的训练数据集，
接下来看看如何使用它来训练一个能 **预测相邻词汇** 的自然语言模型。
<img style="width:800px" src="http://jalammar.github.io/images/word2vec/skipgram-language-model-training.png"></img>

从数据集中的第一个样本开始。
将特征输入到未经训练的模型，让它预测一个可能的相邻单词。
<img style="width:650px" src="http://jalammar.github.io/images/word2vec/skipgram-language-model-training-2.png"></img>
该模型会执行三个步骤并输入预测向量（对应于 **==单词表中每个单词的概率==**)。

在模型训练之前，初始的预测结果肯定是错误的。
**通过训练，要让模型的输出单词的概率尽量吻合 —— 训练集数据中的输出标签**:
<img style="width:550px" src="http://jalammar.github.io/images/word2vec/skipgram-language-model-training-3.png"></img>
**==目标单词概率为1，其他所有单词概率为0==，这样数值组成的向量就是 =="目标向量"==**。

模型的 **输出** 和 **目标向量** 的偏差如何定义？将这两个向量相减，就能得到偏差向量:
<img style="width:700px" src="http://jalammar.github.io/images/word2vec/skipgram-language-model-training-5.png"></img>
这其实就是训练的第一步了。
接下来继续对数据集内下一份样本进行同样的操作，直到遍历所有的样本，就是一轮（epoch）训练了。
通过多轮（epoch）的训练，得到训练好的模型，就可以从模型中提取词嵌入矩阵来用于后续的应用。

以上确实有助于理解整个流程，但这依然不是 word2vec 真正训练的方法。
下面将介绍 word2vec 中一些关键的思想。

#### 3.2.2.1 Negative Sampling 负例采样

> “To attempt an understanding of Muad'Dib without understanding his mortal enemies, the Harkonnens, is to attempt seeing Truth without knowing Falsehood. It is the attempt to see the Light without knowing Darkness. It cannot be.” ~Dune

回想一下上文神经语言模型计算预测值的三个步骤：
<img style="width:550px" src="http://jalammar.github.io/images/word2vec/language-model-expensive.png"></img>
从计算的角度来看，第三步的计算代价非常高，
因为 **需要在数据集中为每个单词求一次概率值**，因此需要寻找一些提高速度表现的方法。

一种方法是将任务分为两个步骤：

- ①、生成高质量的词嵌入（不要担心下一个单词预测）。
- ②、使用这些高质量的词嵌入来训练语言模型（进行下一个单词预测）。

在本文中将专注于第 ① 步（因为这篇文章专注于词嵌入）。
要使用高性能模型生成高质量词嵌入，可以改变预测相邻单词这一任务：
<img style="width:500px" src="http://jalammar.github.io/images/word2vec/predict-neighboring-word.png"></img>

将上述任务切换到一个提取输入与输出单词的模型，并输出一个表明它们是否是邻居的分数
（0表示"不是邻居"，1表示"邻居"）。
<img style="width:500px" src="http://jalammar.github.io/images/word2vec/are-the-words-neighbors.png"></img>
这个简单的变换 ==**将需要的模型从神经网络改为逻辑回归模型**==------因此它变得更简单，计算速度更快。
因此需要相应变化数据集的结构------ ==label 标签值现在是一个值为 0 或 1 的新列==。
它们将全部为1，因为添加的所有单词都是邻居。
<img style="width:600px" src="http://jalammar.github.io/images/word2vec/word2vec-training-dataset.png"></img>
现在的计算速度可谓是神速啦------在几分钟内就能处理数百万个例子。
但是还需要解决一个漏洞。
如果所有的例子都是邻居（目标：1），这个"天才模型"可能会被训练得永远返回1------
准确性是百分百了，但它什么东西都学不到，只会产生垃圾词嵌入结果。
<img style="width:400px" src="http://jalammar.github.io/images/word2vec/word2vec-smartass-model.png"></img>
为了解决这个问题，需要在数据集中引入负样本 - 不是邻居的单词样本。
模型需要为这些样本返回 0。
<img style="width:500px" src="http://jalammar.github.io/images/word2vec/word2vec-negative-sampling.png"></img>
对于数据集中的每个样本，添加了负面示例。它们具有相同的输入字词，标签为0。
但是作为输出词填写什么呢？从词汇表中随机抽取单词
<img style="width:650px" src="http://jalammar.github.io/images/word2vec/word2vec-negative-sampling-2.png"></img>

#### 3.2.2.2 基于负例采样的 Skipgram（SGNS）

现在已经介绍了 word2vec 中的两个（一对）核心思想：**负例采样**，以及 **skipgram**。
<img style="width:700px" src="http://jalammar.github.io/images/word2vec/skipgram-with-negative-sampling.png"></img>

##### 3.2.2.2.1 Word2vec 训练流程

现在已经了解了 skipgram 和负例采样的两个中心思想，下面继续仔细研究 word2vec 的实际训练过程。

在训练开始之前，预先处理训练文本。
在这一步中，首先确定词典的大小（称之为 **`vocab_size`**(样本量)，比如说 10,000）以及哪些词被它包含在内。

在开始训练时，创建两个矩阵 —— Embedding 矩阵和 Context 矩阵。
这两个矩阵在的词汇表中词嵌入了每个单词（所以 `vocab_size` 是他们的维度之一）。
第二个维度是希望每次词嵌入的长度（**`embedding_size`** —— 300是一个常见值，但在前文也看过 50 的例子）。
<img style="width:700px" src="http://jalammar.github.io/images/word2vec/word2vec-embedding-context-matrix.png"></img>
在训练过程开始时，用随机值初始化这些矩阵，然后开始训练。
在每个训练步骤中，采取一个相邻的样本及与其相关的非相邻样本，如下：
<img style="width:700px" src="http://jalammar.github.io/images/word2vec/word2vec-training-example.png"></img>
现在有四个单词（看淡青色的背景颜色）：
输入单词 **`not`** 和输出/上下文单词 : **`thou`**(实际邻居词) 和 (负面例子)**`aaron`** , **`taco`**。

继续查找它们的词嵌入：
**对于输入词，查看 Embedding 矩阵**。
**对于上下文单词，查看 Context 矩阵**（即使两个矩阵都在词汇表中词嵌入了每个单词）。
<img style="width:700px" src="http://jalammar.github.io/images/word2vec/word2vec-lookup-embeddings.png"></img>
然后，**计算输入词嵌入与每个上下文词嵌入的 ==点积==**。
在每种情况下，结果都将是表示输入和上下文词嵌入的相似性的数字。
<img style="width:500px" src="http://jalammar.github.io/images/word2vec/word2vec-training-dot-product.png"></img>
现在需要一种方法将这些分数转化为看起来像概率的东西
需要它们都是正值，并且 处于 0 到 1 之间。
sigmoid 这一逻辑函数正适合用来做这样的事情。

<img style="width:600px" src="http://jalammar.github.io/images/word2vec/word2vec-training-dot-product-sigmoid.png"></img>
现在将 sigmoid 函数的输出作为模型输出。
您可以看到 taco 得分最高， aaron 最低，无论是 sigmoid 操作之前还是之后。

既然未经训练的模型已做出预测，而且确实拥有真实目标标签来作对比，
从目标标签中减去 sigmoid 分数，作为计算模型预测中的 Error (**`loss`**)。
<img style="width:660px" src="http://jalammar.github.io/images/word2vec/word2vec-training-error.png"></img>
`error = target - sigmoid_scores`

这是"机器学习"的"学习"部分。
现在，可以利用这个 loss 误差来调整 `not`、`thou`、`aaron` 和 `taco` 的词嵌入，
使下一次做出这一计算时，结果会更接近目标分数。
<img style="width:660px" src="http://jalammar.github.io/images/word2vec/word2vec-training-update.png"></img>
一次训练步骤就到此结束，
在这一步中得到了训练样本更好一些的词嵌入（`not`、`thou`、`aaron` 和 `taco`）。
现在进行下一步（下一个相邻样本及其相关的非相邻样本），并再次执行相同的过程。
<img style="width:600px" src="http://jalammar.github.io/images/word2vec/word2vec-training-example-2.png"></img>
当循环遍历整个数据集多次时，词嵌入会继续得到改进。
然后就可以停止训练过程，丢弃 Context 矩阵，
并使用 Embeddings 矩阵作为下一项任务的已被训练好的词嵌入。

##### 3.2.2.2.2 窗口大小和负样本数量

word2vec 训练过程中的两个关键超参数是 **窗口大小** 和 **负样本的数量**。
<img style="width:500px" src="http://jalammar.github.io/images/word2vec/word2vec-window-size.png"></img>
不同的任务适合不同的窗口大小。
一种启发式方法是，
- 使用较小的窗口大小（2-15）得到的词嵌入有如下特征：两个词嵌入之间的高相似性得分表明这些单词是可互换的
  （注意，如果只查看附近距离很近的单词，反义词通常可以互换 — 例如，好的和坏的经常出现在类似的语境中）。
- 使用较大的窗口大小（15-50，甚至更多）会得到相似性更能指示单词相关性的词嵌入。

在实际操作中，你通常需要对词嵌入过程 **提供指导** 以帮助读者得到相似的"语感"。
Gensim 默认窗口大小为5（除了输入字本身以外还包括输入字之前与之后的两个字）。

<img style="width:700px" src="http://jalammar.github.io/images/word2vec/word2vec-negative-samples.png"></img>

负样本的数量是训练训练过程的另一个重要因素。
原始论文认为 5-20 个负样本是比较理想的数量，
当数据集足够大时，2-5 个似乎就已经足够了，Gensim 中默认为 5 个负样本。














