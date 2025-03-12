import os
from fastapi import FastAPI, Request, File, UploadFile, Form
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
from langchain.memory import ConversationBufferMemory
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
memory = ConversationBufferMemory(memory_key="history", return_messages=True)


# Defines a route for the home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Defines a route to handle PDF uploads via a POST request
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

    #splits the text into chunks of 500 characters with a 50-character overlap to maintain context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str)]

    #Converts text chunks into vector embeddings using a pre-trained Sentence Transformer model.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_texts(texts, embedding_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})  # Fetch top 2 relevant chunks

    return {"message": "PDF uploaded and processed successfully!"}



# Defines a route to handle user questions via a POST request
@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global retriever, memory

    if retriever is None:
        return {"answer": "No PDF has been uploaded yet. Please upload a PDF first."}

    # Load the LLM
    llm = HuggingFaceEndpoint(
        #repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature=0.1,
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.1,
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
        **Response:** The font size of subheadings is 13.
        Here history is the conversatin\n\n"""

        "{history}\n\n"

        "Give me the answer by consdering the upper conversation and base on the following context\n\n"
        "{context}\n\n"

        "User question: {input}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


    # Create a Question-Answering chain using the retrieved documents and the LLM
    """
    This function constructs a "Stuff" document chain.
    It takes retrieved document chunks and "stuffs" them into a single prompt.
    """
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)



    # Retrieve the current conversation history, if none exists, default to an empty string
    current_history = memory.load_memory_variables({}).get("history", "")

    # Prepare the input dictionary with all required variables
    chain_input = {
        "input": question,
        "history": current_history,
        "context": ""  
    }

    result = rag_chain.invoke(chain_input)
    
    # Save the conversation context using memory
    memory.save_context({"input": question}, {"output": result["answer"]})

    return {"answer": result["answer"]}
