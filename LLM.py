

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
        #content = "\n".join(doc.page_content for doc in context) # 直接使用这个效果貌似更好
        input_ids = self.tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        if len(input_ids) > self.max_token:
            content = self.tokenizer.decode(input_ids[:self.max_token-1])
            warnings.warn("texts have been truncted")
        content = self.prompt_template.format(content, query)
        # print(prompt)
        response, history = self.model.chat(self.tokenizer, content, history=[], **self.kwargs)
        return response

    
