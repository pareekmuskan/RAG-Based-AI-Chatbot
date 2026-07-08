# RAG-Based PDF Chatbot

## Overview

Built a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions in natural language. The chatbot retrieves relevant content from the uploaded documents and generates context-aware responses using Google's Gemini model.

## Features

* Upload and process PDF documents
* Extract and chunk document text
* Generate embeddings and perform semantic search
* Ask questions in natural language
* Receive context-aware answers based on document content

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Google Gemini API
* pdfplumber

## How to Run

1. Clone the repository.
2. Install dependencies:
   pip install -r requirements.txt
3. Add your Gemini API key in a `.env` file.
4. Run:
   streamlit run RAGChatbot.py

## Future Improvements

* Support multiple document formats
* Chat history and conversation memory
* User authentication
* Cloud deployment
