import asyncio

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pathlib import Path
from gemini_webapi import GeminiClient

from rag.src.core.config import CHROMA_DIR, settings

async def main():
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
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    vector_store = Chroma(
        collection_name="markdown_vault",
        embedding_function=embedder_model,
        persist_directory=str(CHROMA_DIR)
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

    #if documents:
    #    vector_store.add_documents(documents)

    print(f"Stored documents in Chroma DB")

    query = str(input("Input query:\n")).strip("\'\"")

    results = vector_store.similarity_search(
        query,
        k=10
    )

    print(f"Received {len(results)} chunks")

    gemini_client = GeminiClient(
        secure_1ps1d=settings.GEMINI_WEB_1PSID,
        secure_1ps1d_ts=settings.GEMINI_WEB_1PSIDTS
    )

    await gemini_client.init(timeout=30, auto_close=False, close_delay=300)

    prompt = f"""
    Using the following context, answer the question below, you may utilise external knowledge but warn the user if done so and why (e.g. insufficient context).

    User's Query: {query}
    
    Context:
    {results}
    """
    response = await gemini_client.generate_content(prompt=prompt, temporary=True)
    print(response.text)



if __name__ == '__main__':
    asyncio.run(main())