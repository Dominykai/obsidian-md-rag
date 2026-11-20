import asyncio

from pathlib import Path
from gemini_webapi import GeminiClient

from rag.src.core.config import get_data_path, settings
from rag.src.models.vault import get_vaults_from_file, Vault, add_vault_to_file, search_vault_for_md_files


async def main():
    vault_path = get_data_path("vaults.json")
    saved_vaults = get_vaults_from_file(vault_path)

    next_free = 0
    for i, vault in enumerate(saved_vaults):
        print(f"{i}. {vault}")
        next_free = i + 1
    print(f"{next_free}. Create new vault")
    select = int(input("Select vault number: "))

    if select == next_free:
        vault_path = Path(str(input("Input vault path: ")))
        add_vault_to_file(vault_path, Vault(path=vault_path))

    else:
        vault_path = saved_vaults[select].path

    md_files = search_vault_for_md_files(vault_path)
    print(f"Discovered {len(md_files)} files")

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