# PDF Reader & Q/A Chatbot

This project is a PDF Reader and Q/A chatbot that allows users to upload a PDF file and ask questions about its content. It uses FastAPI for the backend, LangChain for document processing, FAISS for vector storage, and Hugging Face models for answering user queries.

## Features

- Upload a PDF document.
- Process and split text into meaningful chunks.
- Generate vector embeddings for retrieval.
- Ask questions about the uploaded PDF and receive AI-generated answers.
- Memory-enabled chatbot for contextual conversations.

## Installation & Setup

### 1. Clone the Repository

```sh
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

```sh
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```sh
venv\Scripts\activate
```

**Mac/Linux:**
```sh
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```sh
pip install fastapi uvicorn langchain langchain-community langchain-huggingface faiss-cpu python-dotenv jinja2
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add any required API keys (for Hugging Face, if necessary):

```sh
HUGGINGFACE_API_KEY=<your-api-key>
```

## Running the Project

### 1. Start the FastAPI Server

```sh
uvicorn main:app --reload
```

The server will run at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### 2. Access the Web Interface

Open a browser and go to:

```
http://127.0.0.1:8000/
```

## Usage

### 1. Upload a PDF
- Select a PDF file and click **Upload PDF**.
- The document will be processed and stored for retrieval.

### 2. Ask Questions
- Type a question related to the uploaded PDF in the input box.
- Click **Ask** to receive an AI-generated answer.
- The chatbot retains conversation history for contextual responses.

## Project Structure

```
📂 project-folder/
├── 📄 main.py           # Backend API with FastAPI
├── 📄 index.html        # Frontend web interface
├── 📄 styles.css        # Styling for the frontend
├── 📂 static/           # Static files (CSS, images, etc.)
├── 📂 templates/        # HTML templates
├── 📂 uploaded_files/   # Stores uploaded PDFs
└── 📄 requirements.txt  # Dependencies
```

## Dependencies

- FastAPI
- Uvicorn
- LangChain
- FAISS
- Hugging Face Transformers
- Jinja2
- Python-Dotenv

## Future Enhancements

- Implement a database for persistent storage.
- Support multiple document uploads.
- Improve response generation with more advanced models.

