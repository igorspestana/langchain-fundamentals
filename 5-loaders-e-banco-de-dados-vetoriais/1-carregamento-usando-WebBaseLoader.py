from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(web_paths=["https://www.mpdconsultants.com/"])
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

chunks = splitter.split_documents(docs)

print(f"Total de chunks: {len(chunks)}")

for chunk in chunks:
    print(chunk.page_content)
    print("-" * 50)