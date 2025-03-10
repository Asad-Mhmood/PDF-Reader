from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


# Load the PDF file
loader = PyPDFLoader("G:\VS Code\PDF Reader\SRS Retail Store Surveillance.pdf")
data = loader.load()



# Split data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
docs = text_splitter.split_documents(data)
print("Total number of documents after splitting:", len(docs))




# Extract text content
texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str)]



# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Create FAISS vectorstore
vectorstore = FAISS.from_texts(texts, embedding_model)
print("FAISS Database is created")


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
print('Retriever is set')
print(retriever)



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature=0.1,
    task="text-generation",
    
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "Give me the to the point answer. give very short answer. And to give answer use "
    "the following pieces of retrieved context to answer "
    
  
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

print(prompt)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "what is the font style of document"})
print(response["answer"])