from langchain_text_splitters import RecursiveCharacterTextSplitter
loader=TextLoader("dl-curriculam.pdf")
documents=loader.load()
print(len(documents))
print(documents[0].page_content)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
chunk_overlap=200,
separators=["\n\n", "\n", ".", " "]
)
splitted_docs=text_splitter.split_documents(documents)
print(len(splitted_docs))
print(splitted_docs[0].page_content)