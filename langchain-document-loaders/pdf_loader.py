from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("dl-curriculam.pdf")
documents=loader.load()
print(len(documents))
print(documents[0].page_content)