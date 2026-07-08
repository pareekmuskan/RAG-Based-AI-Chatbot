from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import streamlit as st

# Get API key
GOOGLE_API_KEY = "AQ.Ab8RN6IQ1FB-A5tjN-AaYJQZ0pUD1lmpex7Jc2JNlH8aqznwdQ"

# Streamlit is a Python framework that lets you build interactive web applications
# using only Python—without needing HTML, CSS, or JavaScript.
st.header("My First RAG Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the content of the PDF into text  

if file is not None:
    with pdfplumber.open(file) as pdf:
        text=""
        for page in pdf.pages:
            text+=page.extract_text()+"\n"
    #st.write(text) #to write the extracted text on the webpage

    # split the text into chunks(smaller parts)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," ",""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(text)
    #st.write(chunks)

    # generate embedding using gemini embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
     
    # Store embeddings in vector db

    vector_store=FAISS.from_texts(chunks,embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #generate answer
    #question -> embeddings -> similarity search -> results to LLM -> response


    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    # define the llm and prompts

    # Initialize Gemini Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        max_tokens=1000
    )
    # provide a prompt to llm about how to generate the response

    prompt=ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a pdf document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n\n"
         "Context:\n{context}"),
         ("human","{question}") 
    ])
    
    chain=(
        {"context":retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if user_question:
        response = chain.invoke(user_question)
        st.write(response)        