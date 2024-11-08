# -Generative-LLM-as-Verifiers-
【Generative LLM as Verifiers】推理加速篇：早停法+复用KV缓存+并行推理，实现推理效率提升几十倍

# 1、引言

当前，**判别式任务**在AI项目的下游任务（含推理、训练场景）中，其需求场景、应用领域以及任务体量依旧很大，比如以下场景：

> ![napkin-selection (2).png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731044722854.png?raw=true)

---

> **RAG**检索问答：
> （**意图判别**）判别在每一步中的用户问题是需要联网搜索（检索向量库）还是纯大模型自身问答；
> （**检索路由判别**）如果后端接入了不同的向量库，甚至还需要判别回答指定问题需要检索哪个向量库；
> （**主题标签判别**）预先对文章进行标签生成（判别标签）以期在检索召回阶段优化召回效果；

---

> 大模型**训练数据清洗**：
> 在清洗预训练数据时，结合判别的标签更好地**过滤低质量文本**（有毒性、有害性、低俗色情等等），**提升训练数据质量**；

---

> 大模型**强化训练**：
> **结果监督**：对合成数据进行**偏好打标**（Prefer_answer VS Rejected_answer）以生成后续的偏好对齐训练数据；
> **过程监督**：推理性强的场景下，不仅仅在整篇回答的颗粒度下有一个总的偏好or不偏好的标签（即结果监督），还需对**细化到回答中的每一步的生成结果进行对错标签的判别**（对 or 错），然后再实现基于过程的监督强化学习，这部分可以参阅[Improving mathematical reasoning with process supervision](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/ "Improving mathematical reasoning with process supervision")；

而对于这类判别式任务，生成式大模型在**判别效果层面**已基本可以淘汰掉Bert类模型，Bert类模型还有生存空间的维二优势就是:

1）结构化输出 （对应**生成式大模型对齐结构化输出的能力不完全稳定**）；

2）推理效率 （对应**生成式大模型推理速度慢**）；

而本人之前在[使用生成式大模型的“推断解码”实现文本分类任务的零样本学习&结构化输出](https://zhuanlan.zhihu.com/p/685155140 "使用生成式大模型的“推断解码”实现文本分类任务的零样本学习&结构化输出")这篇博客中提出的方式可以完美解决第一个问题（对齐结构化输出），
![推断解码原理介绍.jpg](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731033283560.jpg?raw=true "推断解码原理")

> 其实该方式和后面deepmind团队发表的[Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/pdf/2408.15240 "Generative Verifiers: Reward Modeling as Next-Token Prediction")该论文里的方式本质是一样的（他们确实很会起标题创造概念：“Generative Verifiers”，这个概念的确很形象，确实就是**验证候选标签的概率**，然后输出判别结果，压缩提炼概念这块值得学习，我们也借鉴该概念用在本篇博客的标题里...）
> ![Generative Verifiers_Reward Modeling as.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731032871010.png?raw=true "Generative Verifiers: Reward Modeling as Next-Token Prediction 原理")
> 只是双方的应用领域不同，deepmind这篇论文中讲述的**Generative Verifiers**的用途就是上文中介绍的在大模型**强化训练**里的**过程监督**领域。
> ![generative llm as verifier 用途_强化训练之过程监督.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731034211354.png?raw=true)

而对于推理效率这块，之前的博客只是简单推广到了**并行推理**，就未再继续深入拓展；
![并行推理提升推理速度.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731035671885.png?raw=true)

**而在本篇博客中，我们就是要将这种Generative LLM as verifiers的方法的推理性能在非硬件层面进行优化，看看如何一步步成功地将推理效率实现几十倍的时效压缩**。

我们将深入探讨一种结合了**并行推理**、**复用KV缓存**和**早停法**的**多任务并行推理**（Multi-Task parellel inference）方法。通过这种策略，我们能够在保证推理精度的同时，显著提升推理速度，实现几十倍的推理时效压缩。
![早停法+KV缓存复用+并行推理.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731044642840.png?raw=true)

这里提到的三个概念（**并行推理**、**复用KV缓存**、**早停法**）及其组合使用，我们都会在后文中详细讲述。

> 精度VS速度；在大部分的项目推理任务中，我们常常面临的更多是在有限的算力资源下最大化推理效率的问题，而非简单追求完美的效果；

> 单纯地追求推理效率，某种程度也可以简单理解为是在**逆verify step by step化**，会牺牲一定的推理效果，需要结合实际的项目场景灵活决策；

# 2、并行推理

并行推理简单理解就是**以空间换时间**（需要加显存资源来实现并行训练或者推理以提效：用上batch_size维度），了解大模型训练过程的对这块概念应该非常熟悉了，这块我们就不再多赘述了。

> 并行推理：对比串行推理，推理空间复杂度从o(1)变为o(n)，时间复杂度从o(n)变为o(1)，牺牲空间换时间

# 3、复用kv缓存(KV Cache)

> 当前大模型推理侧的一个主要优化点就是利用KV缓存这块

KV缓存（Key-Value Cache）是Transformer模型原理中的一种机制，用于存储每一层的注意力机制中的Key和Value张量（可以形象地理解为是在存储上文的信息，将他们压缩在KV cache中），其具体形式大概为：

```python
past_key_values = 
tuple([
(key_tensor_layer_1, value_tensor_layer_1),
(key_tensor_layer_2, value_tensor_layer_2),
...
(key_tensor_layer_n, value_tensor_layer_n)]
)
# 模型的层数： n
# 每层中对应一个key 缓存，一个v的缓存
#每层中每个 k、v缓存的形状：(batch_size, num_heads, sequence_length, head_size)
```

而在生成序列时，因为是自回归式，模型会递归式地**复用上文（前序tokens）对应的KV张量**，然后逐步one by one地生成**新token**（其中推理过程中，也隐含了包括推理**当前token**对应的**query、key、value**）；每生成一个**新token**后，只需再对每一层的Key和Value张量进行填充操作（填充当前**新token**对应的KV张量），循环往复，整个流程**避免了冗余的重复计算**。

> 以一个例子解释：

1) 三星/生产/包括/ -> 手机
   其中，"三星"和"生产" 就是**前序tokens**，"包括"就是**当前token**，"手机"即为推理出的**新token**；/表示分词
   ![KV缓存机制.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731045618739.png?raw=true)
   ......循环递归......
2) 三星/生产/包括/手机 -> 、
   其中，"三星"和"生产"和"包括" 就是**前序tokens**，"手机"就是**当前token**，"、"即为推理出的**新token**
   ...循环递归...

---

> 其中，KV张量的填充细节具体可以参看这份源代码：
> [pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma/blob/main/modeling_gemma.py "pytorch-paligemma")
> ![KV concat.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731046675938.png?raw=true)
> ![KV update picture.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731046773051.png?raw=true)

在本博客中，我们介绍的复用KV缓存的方式具体为expand操作（扩展后的对象与原对象共享内存），可以更有效地压缩显存占用的同时实现并行推理。

```python
# 用于KV缓存复制：为了批量推理，复用多次question_past_kv_cache以并行推理，需要我们扩充kv_cache以对齐形状，具体通过expand函数实现，注意因为question_past_kv_cache为元组数据(不可变的数据结构据)，不便直接更改，此处选择新建一个kv缓存，注意这并不会增加显存占用
    def kv_cache_expand(self, expand_num: int, past_kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        扩展 KV 缓存以支持批量推理。

        参数:
        - expand_num: 扩展的批量大小倍数。
        - past_kv_cache: 输入的 KV 缓存，是一个元组，其中每个元素是一个包含两个张量的元组（分别表示 Key 和 Value）。

        返回:
        - 扩展后的 KV 缓存，形状为 (batch_size * expand_num, num_heads, sequence_length, head_size)。
        """
        if not past_kv_cache:
            raise ValueError("past_kv_cache 不能为空")

        # 获取 KV 缓存的形状信息
        batch_size = past_kv_cache[0][0].shape[0]
        num_heads = past_kv_cache[0][0].shape[1]
        sequence_length = past_kv_cache[0][0].shape[2]
        head_size = past_kv_cache[0][0].shape[3]

        # 创建新的 KV 缓存
        expanded_kv_cache = []
        for layer_kv_tuple in past_kv_cache:
            if len(layer_kv_tuple) != 2:
                raise ValueError("每个层的 KV 缓存应包含两个张量（Key 和 Value）")

            key_tensor, value_tensor = layer_kv_tuple
            if key_tensor.shape != value_tensor.shape:
                raise ValueError("Key 和 Value 张量的形状应一致")

            expanded_key = key_tensor.expand(batch_size * expand_num, num_heads, sequence_length, head_size)
            expanded_value = value_tensor.expand(batch_size * expand_num, num_heads, sequence_length, head_size)
            expanded_kv_cache.append((expanded_key, expanded_value))

        return tuple(expanded_kv_cache)
```

![扩展kv环节以支持批量推理.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731051798434.png?raw=true)

# 4、早停法

大家一听到**早停法**这个概念，可能首先想到的是一种在机器学习和深度学习中训练过程中的常用优化策略，用于**防止模型过拟合**。其基本思想是在训练过程中，监控验证集上的性能，并在验证集性能不再提升时**提前停止训练**，从而避免模型在训练集上过度拟合。

在本文中，我们介绍的早停法也是把这个概念迁移应用到推理侧；

具体来说，是在**验证候选答案概率**时，**只考虑验证每个候选答案中的第一个token的概率**，从而大幅减少推理耗时。

> 可以将推理的时间复杂度或者空间复杂度从**o(n)压缩为o(1)**；
> 如果对比的是串行推理，那么就是将时间复杂度从o(n)压缩为o(1)；
> 如果对比的是并行推理，那么就是将空间复杂度从o(n)压缩为o(1)；
> 总体表现为：既节省了显存占用（空间复杂度），又节省了时间占用（时间复杂度）。

这种方法基于以下假设：

- 如果多个候选答案的**第一个token**完全不同，那么可以通过**只考虑每个候选答案中第一个token的生成概率**来代替**每个候选答案对应的概率**；

> 这背后其实隐含了假设：第一个token的概率分布能够较好地反映整个候选答案序列的概率分布。
> 从一个提示词例子代入下：

```txt
请判断中国位于以下哪个大洲：
a) 大洋洲
b) 北美洲
c) 亚洲
d) 南极洲
```

我们候选的答案标签为：

```python
['a) 大洋洲', 'b) 北美洲', 'c) 亚洲', 'd) 南极洲']
```

当我们把这个提示词送给大模型推理时，**如果下一个token大模型输出的是c，其实你就可以提前结束推理了**，因为你看到**c**就知道答案是亚洲了（虽然一般来说，生成式的大模型要等到'eos'这个special token输出后才结束推理）；对应于概率角度，即在提供了提示词给大模型推理后，大模型在预测下一个潜在token的概率分布假设为：

```python
{'a': 0.015, 'b': 0.02, 'c': 0.95, 'd': 0.005}  # 其他：0.01
```

我们就把上面的每个候选答案的第一个token的预测概率分布当做最终的候选答案的预测概率分布

```python
{'a) 大洋洲': 0.015, 'b) 北美洲': 0.02, 'c) 亚洲': 0.95, 'd) 南极洲': 0.005}  # 其他：0.01
```

通过这种提前结束推理标签概率的方式，我们相当于将n个候选标签的概率推理任务的时间/空间复杂度压缩为了o(1)，和bert类的判别式模型的最后那个判别头的推理的时间复杂度一样
![推理中的早停法.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731050715863.png?raw=true)

**早停法的局限性**

早停法的有效性依赖于**每个候选答案的第一个token的概率分布能够较好地反映整个候选答案序列的概率分布**的假设合理性。

- 如果每个候选答案的第一个token的概率分布不能很好地反映整个序列的概率分布，早停法的效果可能会受到影响（不过一般偏差不大）；
- 如果候选答案的第一个token存在重复的情况时，我们就要**重构造答案标签以使得每个候选答案的第一个token完全不同**。

# 5、早停法+复用KV缓存+并行推理，实现几十倍的推理加速

在NLP下游任务中，我们有时需要对同一篇文本（或图片）执行多项任务，

> 举例：

```python
article = "在北京时间3月3日的NBA常规赛中，洛杉矶湖人队与丹佛掘金队的比赛成为了篮球历史上的一个重要时刻......"
question = 
       ['请判断这篇文章最适合被划归到以下哪个类目结果，请直接输出类目结果，不要解释：\n财经，政治，娱乐，体育，军事，科技，农业民生，能源，汽车，创新设计，环境，其它',
       '这篇新闻提到的联赛最可能来源于哪个国家：\n英国，中国，法国，日本，美国，印度\n请直接回复最可能的语种，不要解释',
       '这篇新闻是否提到了色情等有毒信息，请直接回复是或否，不要解释',
       '这篇新闻是否是明显的广告文案，请直接回复是或否，不要解释']
answer_label = 
[['财经', '政治', '娱乐', '体育', '军事', '科技', '农业民生', '能源', '汽车', '创新设计', '环境', '其它'],
['英国', '中国', '法国', '日本', '美国', '印度'],
['是', '否'],
['是', '否']]
task = {k:v for k,v in zip(question, answer_label)}
```

那么我们就可以将上文中讲述的**早停法+复用KV缓存+并行推理**统用在一块；

- 一方面，多个任务其实是在**识别处理同一篇文章（图片）**，这里面有**复用KV缓存的优化空间**（主要就是指复用文章/图片的KV缓存，文章越长（图片像素越大），优化的空间越大）；
- 另一方面，每个任务中的的多候选标签的概率推理我们可以利用**早停法**进行推理加速：空间/时间复杂度o(1)；
- 最后，将多个任务的推理并行化（并行推理），需要特别指出因为任务中的问题部分的token的长度一般不大，而**KV缓存的主力是文章/图片**，而KV缓存扩展时因为expand的操作不会新增显存占用，所以并行推理时，显存的占用量增加不会非常夸张；

我们实地实验测试一下耗时（实验环境：A30卡，模型：Qwen2-1.5B-Instruct），具体如下：

- 1、使用原始的串行推理模式执行**单任务**的候选答案的概率验证推理：
  ![耗时对比-原始模式.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731053394664.png?raw=true)
  如果扩展到8个任务的话，总耗时就是**8秒**左右
- 2、早停法+复用KV缓存+并行推理，执行**多任务并行**的概率验证推理：
  ![耗时对比-多任务并行推理.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731053485228.png?raw=true)
  耗时被压缩到了**0.15秒**，同时显存占用仅增加了一倍多一点(batch_size由1变为8）

我们做到了，实现几十倍的推理加速！

# 6、后续

本文未继续将这种方法进一步推广应用到基于**VLM**架构的多模态模型（即底层还是decoder-only的模型方案）
![VLM architecture.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-11-08/1731054865450.png?raw=true)

关于当前的多模态模型的架构路线介绍，可以阅读这篇文章：[Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms "Understanding Multimodal LLMs")

但是，这套方法完全是可以复用的，而且更适合于多模态模型，因为**图片的KV缓存对应的token的数量一般都挺大的**，本文的这套法子非常适配，有很大的加速空间，感兴趣的可以自行尝试进行迁移复用。

# 7、实验脚本
感谢你的阅读，本博客全部实验脚本请参看:[【Generative LLM as Verifiers】推理加速篇：早停法+复用KV缓存+并行推理，实现推理效率提升几十倍.ipynb](https://github.com/Paul33333/-Generative-LLM-as-Verifiers-/blob/main/%E3%80%90Generative_LLM_as_Verifiers%E3%80%91%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E7%AF%87%EF%BC%9A%E6%97%A9%E5%81%9C%E6%B3%95%2B%E5%A4%8D%E7%94%A8KV%E7%BC%93%E5%AD%98%2B%E5%B9%B6%E8%A1%8C%E6%8E%A8%E7%90%86%EF%BC%8C%E5%AE%9E%E7%8E%B0%E6%8E%A8%E7%90%86%E6%95%88%E7%8E%87%E6%8F%90%E5%8D%87%E5%87%A0%E5%8D%81%E5%80%8D.ipynb)
