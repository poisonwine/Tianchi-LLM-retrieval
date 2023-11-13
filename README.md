# Tianchi-LLM-retrieval
2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答， 76+ baseline

## 方案简介
1.LLM采用 internlm-7b-chat、Qwen-7b-chat和chatglm3-6b答案融合，这三个目前应该是开源模型中10B以下性能最优的。（把其中任意一个模型换成gpt-4，可以直接到79+....）

2.embedding 模型采用 BGE-large-zh-v1.5, 其它项目用下来这个开源embedding模型效果最好，top5召回和openai embedding模型差不多

3.pdf 解析使用pypdf2， 采用spacy分词，每个chunk近似切成256（可能大于256），pdf解析时去掉了目录，合并了一些简短的上下文

4.利用FAISS语义检索top5

## 可改进的地方(仅供参考)
1. pdf 解析目前相对来说比较粗糙，可改进的点还是很多的，这也是效果提升的主要来源

   - 细粒度和完整上下文之间的平衡可以考虑改进，类似langchain ParentDocumentRetriever，小块的metadata存的是大块id，检索小块，利用id合并上下文
   - 尽量去掉一些解析不正常的特殊符号啥的，保持语义的的连贯性，毕竟开源的模型理解能力有限
   - 其它的一些解析方式，如按照目录解析，层级结构
  
2. 召回: 目前只使用了FAISS语义召回，可以尝试多种召回方式（BM25）+ rerank
   
3. prompt工程，你懂的，包含提示词和召回材料的组织格式，不同的提示词差别还是有挺大的
   
4. BGE embedding模型可以微调，从其它项目来看是会有提升的，只是数据集的构造可能需要借助能力更强的大模型，同理LLM，可以instruct QA 微调。建议精力不多的还是放在前三种方案
   

