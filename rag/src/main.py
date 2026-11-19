from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

def main():
    vault_path = Path(str(input("Input path to vault:\n")).strip("\'\""))
    md_files = list(vault_path.rglob("*.md"))
    print(f"Discovered {len(md_files)} files")

    for md_file in md_files:
        body = md_file.read_text()

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"),
                ("####", "H4"), ("#####", "H5"), ("######", "H6"),
            ],
            strip_headers=False
        )
        chunks = splitter.split_text(body)

        chunk_size = 500
        fallback = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        final = []
        for c in chunks:
            if len(c.page_content) > chunk_size:
                final.extend(fallback.split_documents([c]))
            else:
                final.append(c)





if __name__ == '__main__':
    main()