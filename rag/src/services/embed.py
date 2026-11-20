from pathlib import Path
from typing import Iterable

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedder_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
)

vector_store = Chroma(
    collection_name="markdown_vault",
    embedding_function=embedder_model,
    persist_directory=str(get_data_path("chroma"))
)

def chunk_file(file_path: Path, splitter = None):
    if not splitter:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"),
                ("####", "H4"), ("#####", "H5"), ("######", "H6"),
            ],
            strip_headers=False
        )

    body = file_path.read_text()
    return splitter.split_text(body)

def chunk_document(documents: Iterable, splitter = None):
    if not splitter:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)