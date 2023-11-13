import json

from LLM import LLMPredictor
from embeddings import BGEpeftEmbedding
from langchain import FAISS
from pdfparser import extract_page_text


def main():
    filepath = "./data/初赛训练数据集.pdf"
    from pdfparser import extract_page_text
    docs = extract_page_text(filepath=filepath, max_len=256)
    model1 = "./base_model/internlm-7b-chat"
    model2 = "./base_model/Qwen-7b-chat"
    model3 =  "./base_model/chatglm3-6b-chat"
    embedding_path ="./base_model/bge-large-zh-v1.5"
    llm1 = LLMPredictor(model_path=model1, is_chatglm=False, device='cuda:0')
    llm2 = LLMPredictor(model_path=model2, is_chatglm=False, device='cuda:1')
    llm3 = LLMPredictor(model_path=model3, is_chatglm=True, device='cuda:2')
    embedding_model = BGEpeftEmbedding(model_path=embedding_path)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(folder_path='./vector', index_name='index_256')
    # db = FAISS.load_local(folder_path='./vectors', index_name='index', embeddings=embeddings)
    result_list = []

    with open('./data/test.json', 'r', encoding='utf-8') as f:
        result = json.load(f)

    for i, line in enumerate(result):
        print(f"question {i}:", line['question'])

        search_docs = db.similarity_search(line['question'], k=5)
        res1 = llm1.predict(search_docs, line['question'])
        res2 = llm2.predict(search_docs, line['question'])
        res3 = llm3.predict(search_docs, line['question'])
        print('\n')
        line['answer_1'] = res1
        line['answer_2'] = res2
        line['answer_3'] = res3
        result_list.append(line)
    

    with open('./data/submit.json', 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
