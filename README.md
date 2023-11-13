# Tianchi-LLM-retrieval
2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答， 75+ baseline

## 方案简介
1.LLM采用 internlm-7b-chat、Qwen-7b-chat和chatglm3-6b答案融合，这三个目前应该是开源模型中10B以下性能最优的

2.embedding 模型采用 BGE-large-zh-v1.5, 其它项目用下来这个开源embedding模型效果最好，top5召回和openai embedding模型差不多

3.pdf 解析使用pypdf2， 简单的按chunk=256切，去掉了目录，合并了一些简短的上下文，利用FAISS语义检索top5

