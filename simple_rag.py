


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings


llm = OllamaLLM(model ='gemma3:1b')
embeddings = OllamaEmbeddings(model = 'nomic-embed-text:latest')

file_path = 'Python for Everyone1.pdf'
loader = PyPDFLoader(file_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=50)

splits = text_splitter.split_documents(pages)

persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents = splits,
    embedding = embeddings,
    persist_directory = persist_directory
)

question = ' what is python?'

docs = vectordb.similarity_search(question,k=3)

template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_Prompt = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectordb.as_retriever(),
    return_source_documents = True,
    chain_type_kwargs ={'prompt':QA_CHAIN_Prompt}

)

result = qa_chain({'query':question})
print(result['result'])
