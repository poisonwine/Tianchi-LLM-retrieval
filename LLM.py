

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from peft import PeftModel
from langchain.schema.embeddings import Embeddings
from typing import List
import warnings

def build_template():
    prompt_template = "你是一个汽车驾驶安全员,精通有关汽车驾驶、维修和保养的相关知识。请你基于以下汽车手册材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。\n" \
                        "以下是材料：\n---" \
                        "{}\n" \
                        "用户问题：\n" \
                        "{}\n" 
    return prompt_template

class LLMPredictor(object):
    def __init__(self, model_path, adapter_path=None, is_chatglm=False, device="cuda", **kwargs):
        
        if is_chatglm:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained( model_path,
                                                        trust_remote_code=True,
                                                        use_fast=False if self.model.config.model_type == 'llama' else True,
                                                        padding_side='left')
     
        self.max_token = 4096
        self.prompt_template = build_template()
        self.kwargs = kwargs
        self.device = torch.device(device)
        self.model.eval()
        self.model.to(self.device)
        print('successful  load LLM', model_path)


    def predict(self, context, query):
        # context List [doc]
        # query str
        content = ""
        for i, doc in enumerate(context):
            content +=  doc.page_content + "\n---\n"
        input_ids = self.tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            content = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        content = self.prompt_template.format(content, query)
        # print(prompt)
        response, history = self.model.chat(self.tokenizer, content, history=[], **self.kwargs)
        return response

class BGEpeftEmbedding(Embeddings):
    def __init__(self, model_path, lora_path=None, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer= AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path).eval()
            print('merged embedding model')
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.batch_size = batch_size

        self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

        print("successful load embedding model")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        num_texts = len(texts)

        sentence_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]

            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input.to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                batch_embeddings = model_output[0][:, 0]
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

        return sentence_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True, truncation=True, return_tensors='pt')
        encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()
    
