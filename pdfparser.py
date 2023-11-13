import os
from typing import List
import re
import tqdm

from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.schema import Document
import spacy
import PyPDF2


class ChinesePDFTextLoader(UnstructuredFileLoader):

    def _get_elements(self) -> List:

        def extract_text_from_pdf(filepath, dir_path="tmp_files"):
            """Extract text content from a PDF file."""
            contents = []
            import PyPDF2
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text().strip()
                    raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                    raw_text = [re.sub(r"\s+", " ", text) for text in raw_text]
                    raw_text = [re.sub(r'\.{2,}', " ", text) for text in raw_text]
                    new_text = ''
                    for text in raw_text:
                        new_text += text
                        if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》',
                                        '」', '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                            contents.append(Document(page_content=new_text, metadata={'page_number': i+1}))
                            # contents.append(new_text)
                            new_text = ''
                    if new_text:
                        contents.append(Document(page_content=new_text, metadata={'page_number': i+1}))
                        # contents.append(new_text)
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            txt_file_path = os.path.join(full_dir_path, f"{os.path.split(filepath)[-1]}.txt")
            print(" txt file have been stored in {}".format(txt_file_path))
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for doc in contents:
                    fout.write(str(doc))
                    fout.write("\n")
                fout.close()
            return txt_file_path

        txt_file_path = extract_text_from_pdf(self.file_path)
        from utils import PartitionChineseText
        return PartitionChineseText(filename=txt_file_path, **self.unstructured_kwargs)
    

def extract_page_text(filepath, max_len=256):
    page_content  = []
    spliter = spacy.load("zh_core_web_sm")
    chunks = []
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        page_count = 10
        pattern = r'^\d{1,3}'
        for page in tqdm.tqdm(pdf_reader.pages[page_count:]):
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.split('\n')]
            new_text = '\n'.join(raw_text[1:])
            new_text = re.sub(pattern, '', new_text).strip()
            page_content.append(new_text)
            max_chunk_length = max_len  # 最大 chunk 长度

            current_chunk = ""
            if len(new_text) > 10:
                for sentence in spliter(new_text).sents:
                    sentence_text = sentence.text
                    if len(current_chunk) + len(sentence_text) <= max_chunk_length:
                        current_chunk += sentence_text
                    else:
                        chunks.append(Document(page_content=current_chunk, metadata={'page':page_count+1}))
                        current_chunk = sentence_text
                # 添加最后一个 chunk（如果有的话）
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk, metadata={'page':page_count+1}))
            page_count += 1
    cleaned_chunks = []
    i = 0
    while i <= len(chunks)-2: #简单合并一些上下文
        current_chunk = chunks[i]
        next_chunk = chunks[min(i+1, len(chunks)-1)]
        if len(next_chunk.page_content) < 0.5 * len(current_chunk.page_content):
            new_chunk = Document(page_content=current_chunk.page_content + next_chunk.page_content, metadata=current_chunk.metadata)
            cleaned_chunks.append(new_chunk)
            i += 2
        else:
            i+=1
            cleaned_chunks.append(current_chunk)

    return cleaned_chunks
