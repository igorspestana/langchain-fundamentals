from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./a-practical-guide-to-building-with-gpt-5.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

chunks = splitter.split_documents(docs)

print(f"Total de chunks: {len(chunks)}")