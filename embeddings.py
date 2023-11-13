from transformers import AutoTokenizer,  AutoModel
import torch
from peft import PeftModel
from langchain.schema.embeddings import Embeddings
from typing import List

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
    
