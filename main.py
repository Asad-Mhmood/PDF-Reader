import os
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates
templates = Jinja2Templates(directory="templates")

# Global variables to store processed data
vectorstore = None
retriever = None

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, retriever

    # Save uploaded file
    file_path = f"uploaded_files/{file.filename}"
    os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Load and process PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
    docs = text_splitter.split_documents(data)

    texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str)]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_texts(texts, embedding_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    return {"message": "PDF uploaded and processed successfully!"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global retriever

    if retriever is None:
        return {"answer": "No PDF has been uploaded yet. Please upload a PDF first."}

    # Load the LLM
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        temperature=0.1,
        task="text-generation",
    )

    system_prompt = (
    """You are a helpful assistant. Answer all questions in a concise and direct manner.

Guidelines for responses:
- Do NOT include labels like "Human:", "AI:", "Machine:", etc.
- Do NOT reformat answers unless explicitly requested.
- Answer to the point without repeating the question.
- Use proper sentence structure (e.g., "The font size of headings is 14.").
- If asked for a list, provide a clean bullet-point format without unnecessary text.

Avoid:
- Adding unnecessary dialogue formatting.
- Repeating the input question in the response.
- Providing excessive or unrelated details.

Example:
**Question:** What is the font size of subheadings?
**Response:** The font size of subheadings is 13."""
    
  
    "{context}"
)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print('Before Query in Processing')
    response = rag_chain.invoke({"input": question})
    print('Query in Processing')

    return {"answer": response["answer"]}
