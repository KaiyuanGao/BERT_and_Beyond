# ERNIE_THU



![](https://gblobscdn.gitbook.com/assets%2F-MMs7dhj84KIv2oyXzeq%2F-MMtYbVQ7QGInLfFmvj-%2F-MMuCEn82Z1uVZsEDLfQ%2Fimage.png?alt=media&token=e7a737da-f964-4540-a06c-7241f6f1979a)



- 论文：ERNIE: Enhanced Language Representation with Informative Entities
- 地址：https://arxiv.org/pdf/1905.07129.pdf
- 源码：https://github.com/thunlp/ERNIE



本文的工作也是属于对BERT锦上添花，将知识图谱的一些结构化信息融入到BERT中，使其更好地对真实世界进行语义建模。也就是说，原始的bert模型只是机械化地去学习语言相关的“合理性”，而并学习不到语言之间的语义联系，打个比喻，就比如掉包xia只会掉包，而不懂每个包里面具体是什么含义。于是，作者们的工作就是如何将这些额外的知识告诉bert模型，而让它更好地适用于NLP任务。

但是要将外部知识融入到模型中，又存在两个问题：
- **Structured Knowledge Encoding:**  对于给定的文本，如何高效地抽取并编码对应的知识图谱事实；
- **Heterogeneous Information Fusion:** 语言表征的预训练过程和知识表征过程有很大的不同，它们会产生两个独立的向量空间。因此，如何设计一个特殊的预训练目标，以融合词汇、句法和知识信息又是另外一个难题。

为此，作者们提出了ERNIE模型，同时在大规模语料库和知识图谱上预训练语言模型：
1. **抽取+编码知识信息：** 识别文本中的实体，并将这些实体与知识图谱中已存在的实体进行实体对齐，具体做法是采用知识嵌入算法（如TransE），并将得到的entity embedding作为ERNIE模型的输入。基于文本和知识图谱的对齐，ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。
2. **语言模型训练：**  在训练语言模型时，除了采用bert的MLM和NSP，另外随机mask掉了一些实体并要求模型从知识图谱中找出正确的实体进行对齐（这一点跟baidu的entity-masking有点像）。

okay，接下来看看模型到底长啥样？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603210332777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)
如上图，整个模型主要由两个子模块组成：
- 底层的**textual encoder (T-Encoder)**，用于提取输入的基础词法和句法信息，N个；
- 高层的**knowledgeable encoder (K-Encoder)**， 用于将外部的知识图谱的信息融入到模型中，M个。
#### knowledgeable encoder
这里T-encooder跟bert一样就不再赘述，主要是将文本输入的三个embedding加和后送入双向Transformer提取词法和句法信息：$$\left\{\boldsymbol{w}_{1}, \ldots, \boldsymbol{w}_{n}\right\}=\mathrm{T}-\text { Encoder }\left(\left\{w_{1}, \ldots, w_{n}\right\}\right)$$

K-encoder中的模型称为aggregator，输入分为两部分：
- 一部分是底层T-encoder的输出$\left\{\boldsymbol{w}_{1}, \ldots, \boldsymbol{w}_{n}\right\}$
- 一部分是利用TransE算法得到的文本中entity embedding，$\left\{e_{1}, \dots, e_{m}\right\}$
- 注意以上为第一层aggregator的输入，后续第K层的输入为第K-1层aggregator的输出

接着利用multi-head self-attention对文本和实体分别处理：$$\begin{aligned}\left\{\tilde{\boldsymbol{w}}_{1}^{(i)}, \ldots, \tilde{\boldsymbol{w}}_{n}^{(i)}\right\} &=\mathrm{MH}-\operatorname{ATT}\left(\left\{\boldsymbol{w}_{1}^{(i-1)}, \ldots, \boldsymbol{w}_{n}^{(i-1)}\right\}\right) \\\left\{\tilde{\boldsymbol{e}}_{1}^{(i)}, \ldots, \tilde{\boldsymbol{e}}_{m}^{(i)}\right\} &=\mathrm{MH}-\operatorname{ATT}\left(\left\{\boldsymbol{e}_{1}^{(i-1)}, \ldots, \boldsymbol{e}_{m}^{(i-1)}\right\}\right) \end{aligned}$$
然后就是将实体信息和文本信息进行融合，实体对齐函数为$e_{k}=f\left(w_{j}\right)$:
- 对于有对应实体的输入：
$$\begin{aligned} \boldsymbol{h}_{j} &=\sigma\left(\tilde{\boldsymbol{W}}_{t}^{(i)} \tilde{\boldsymbol{w}}_{j}^{(i)}+\tilde{\boldsymbol{W}}_{e}^{(i)} \tilde{\boldsymbol{e}}_{k}^{(i)}+\tilde{\boldsymbol{b}}^{(i)}\right) \\ \boldsymbol{w}_{j}^{(i)} &=\sigma\left(\boldsymbol{W}_{t}^{(i)} \boldsymbol{h}_{j}+\boldsymbol{b}_{t}^{(i)}\right) \\ \boldsymbol{e}_{k}^{(i)} &=\sigma\left(\boldsymbol{W}_{e}^{(i)} \boldsymbol{h}_{j}+\boldsymbol{b}_{e}^{(i)}\right) \end{aligned}$$
- 对于没有对应实体的输入词：
$$\begin{aligned} \boldsymbol{h}_{j} &=\sigma\left(\boldsymbol{\boldsymbol { W }}_{t}^{(i)} \tilde{\boldsymbol{w}}_{j}^{(i)}+\tilde{\boldsymbol{b}}^{(i)}\right) \\ \boldsymbol{w}_{j}^{(i)} &=\sigma\left(\boldsymbol{W}_{t}^{(i)} \boldsymbol{h}_{j}+\boldsymbol{b}_{t}^{(i)}\right) \end{aligned}$$

上述过程就是一个aggregator的操作，整个K-encoder会叠加M个这样的block：$$\left\{\boldsymbol{w}_{1}^{(i)}, \ldots, \boldsymbol{w}_{n}^{(i)}\right\},\left\{e_{1}^{(i)}, \ldots, \boldsymbol{e}_{m}^{(i)}\right\}=\text { Aggregator }(\left\{\boldsymbol{w}_{1}^{(i-1)}, \ldots, \boldsymbol{w}_{n}^{(i-1)}\right\},\left\{e_{1}^{(i-1)}, \ldots, e_{m}^{(i-1)}\right\} )$$
最终的输出为最顶层的Aggregator的token embedding和entity embedding。
#### 改进的预训练
除了跟bert一样的MLM和NSP预训练任务，本文还提出了另外一种适用于信息融合的预训练方式，**denoising entity auto-encoder (dEA).** 跟baidu的还是有点不一样，这里是有对齐后的entity sequence输入的，而百度的是直接去学习entity embedding。dEA 的目的就是要求模型能够根据给定的实体序列和文本序列来预测对应的实体：$$p\left(e_{j} | w_{i}\right)=\frac{\exp \left(1 \text { inear }\left(\boldsymbol{w}_{i}^{o}\right) \cdot \boldsymbol{e}_{j}\right)}{\sum_{k=1}^{m} \exp \left(1 \text { i near }\left(\boldsymbol{w}_{i}^{o}\right) \cdot \boldsymbol{e}_{k}\right)}$$
#### 微调
为了使得模型可以更广泛地适用于不同的NLP任务，作者也学习BERT设计了不同的特殊的token：
- 【CLS】：该token含有句子信息的表示，可适用于一般任务

- 【HD】和【TL】：该token表示关系分类任务中的头实体和尾实体（类似于传统关系分类模型中的位置向量），然后使用【CLS】来做分类；

- 【ENT】：该token表示实体类型，用于entity typing等任务。

  

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019060321525443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)

  试验部分也略过了哈~感觉有些部分还不是很清晰，需要看看源码...
##### reference
- [ACL 2019将会有哪些值得关注的论文？](https://www.zhihu.com/question/324223170/answer/686289852)
- [ACL 2019 | 基于知识增强的语言表示模型，多项NLP任务表现超越BERT](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247497535&idx=1&sn=41565f76368c028cf6cc9204e6f36384&chksm=96ea28bfa19da1a945923fceca0351c50ffa99958a57eced44ea19ef14fbbe3fcfb519b49fe1&scene=0&xtrack=1&key=5e1a44c6b19fdb90d4f39f57aad794cb225ae4b125c74547f68efd27e4a440ed12133aa3846f3246a59c3300e60af78eac543952b7f66291b3bb98aabab234a5c4cdbf011e018ccd51c55ad70dcbc145&ascene=1&uin=MTA1NDIwMzgyMQ==&devicetype=Windows%2010&version=62060833&lang=zh_CN&pass_ticket=GXxtNBoDpN/xpinrCr5v68DE8xs9w5fNjEzknaTiKJSZER4aMw4zfhObcR/hFHhL)
- [ACL 2019 | 清华等提出ERNIE：知识图谱结合BERT才是「有文化」的语言模型](https://www.jiqizhixin.com/articles/2019-05-26-4)
- [官方源码](https://github.com/thunlp/ERNIE)