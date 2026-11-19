from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = ROOT_DIR / "rag" / "data" / "chroma.db"

def main():
    vault_path = Path(str(input("Input path to vault:\n")).strip("\'\""))
    md_files = list(vault_path.rglob("*.md"))
    print(f"Discovered {len(md_files)} files")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"), ("##", "H2"), ("###", "H3"),
            ("####", "H4"), ("#####", "H5"), ("######", "H6"),
        ],
        strip_headers=False
    )

    chunk_size = 500
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)

    embedder_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-4B",
    )

    vector_store = Chroma(
        collection_name="markdown_vault",
        embedding_function=embedder_model,
        persist_directory=str(CHROMA_DB_PATH)
    )

    documents = []

    for md_file in md_files:
        body = md_file.read_text()
        chunks = splitter.split_text(body)
        final = []
        for c in chunks:
            if len(c.page_content) > chunk_size:
                final.extend(fallback_splitter.split_documents([c]))
            else:
                final.append(c)

        for chunk in final:
            chunk.metadata["source"] = str(md_file.relative_to(vault_path))

        documents.extend(final)

    print(f"Generated {len(documents)} chunks")

    if documents:
        vector_store.add_documents(documents)

    print(f"Stored documents in Chroma DB")

    results = vector_store.similarity_search(
        "How do Regression Trees work?",
        k=10
    )

    for result in results:
        print(result.page_content)
        print(result.metadata)
        print("---")




if __name__ == '__main__':
    main()