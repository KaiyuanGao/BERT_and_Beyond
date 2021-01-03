# Roberta
![](https://gblobscdn.gitbook.com/assets%2F-MMs7dhj84KIv2oyXzeq%2F-MMxR0qTZO639FG_g4_X%2F-MMxRCNPRIIr1zBQ-Mim%2Fimage.png?alt=media&token=97cedc9f-1dd4-40de-975d-241038c00ce4)

- 论文：RoBERTa: A Robustly Optimized BERT Pretraining Approach
- 地址：https://arxiv.org/pdf/1907.11692.pdf
- 源码：https://github.com/pytorch/fairseq



‌

看了一眼RoBERTa的作者和前面SpanBERT的作者基本都是一样的...可以...这都不是重点！重点是-----XLNet屠榜了，BERT坐不住了，文章指出BERT是完完全全的**underfit**，于是他们又对BERT进行了一次改造计划，当然，最终结果又是：屠榜。恭喜BERT重回榜首  :)

> Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods.



整理了一下RoBERTa相比原始BERT模型的新的配方：

- 使用更大的预训练语料（BERT为16G，RoBERTa直接到了160G）
- 更长的训练时间：100k to 300k steps
- 更大的batch：2k to 8k
- 丢弃了NSP任务
- 使用**full-length sequences**，而不是截断的文本
- 修改**static masking**策略为**dynamic masking**
- 优化器参数调整



![](https://img-blog.csdnimg.cn/20190811101134402.png?x-oss-process=image%2Fwatermark%2Ctype_ZmFuZ3poZW5naGVpdGk%2Cshadow_10%2Ctext_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ%3D%3D%2Csize_16%2Ccolor_FFFFFF%2Ct_70#pic_center)



### **动态掩码**

在原始的BERT实现中，mask操作是在数据预处理的时候就完成的，这样在每个训练epoch中数据的mask位置都是相同的，这显然是不太合适的。而动态mask则是对于每一个输入都生成一次新的mask，这对于更大训练数据集/更大训练步数是很重要的。实验结果如下，dynamic masking效果是比static好了一点点，但是四舍五入等于一样...

![](https://img-blog.csdnimg.cn/20190811095525485.png?x-oss-process=image%2Fwatermark%2Ctype_ZmFuZ3poZW5naGVpdGk%2Cshadow_10%2Ctext_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ%3D%3D%2Csize_16%2Ccolor_FFFFFF%2Ct_70#pic_center)



### **NSP任务**

同XLNet/SpanBERT一样，作者在这里也是发现NSP任务对下游任务并不会起到帮助甚至有点小危害，

![](https://img-blog.csdnimg.cn/20190811100209168.png?x-oss-process=image%2Fwatermark%2Ctype_ZmFuZ3poZW5naGVpdGk%2Cshadow_10%2Ctext_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ%3D%3D%2Csize_16%2Ccolor_FFFFFF%2Ct_70#pic_center)

‌

### **reference**

- [官方开源代码](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
- [重回榜首的BERT改进版开源了，千块V100、160GB纯文本的大模型](https://zhuanlan.zhihu.com/p/75899781)
- [如何评价RoBERTa?](https://www.zhihu.com/question/337776337/answer/768731809)
