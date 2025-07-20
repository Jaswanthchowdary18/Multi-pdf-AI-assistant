import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # Reverting to PyPDF2 for stability
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from htmlTemplates import css, bot_template, user_template

CHUNK_SIZE = 800  
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
LLM_MODEL = "google/flan-t5-small"  
MAX_TOKENS = 256  

@st.cache_data(show_spinner=False, max_entries=3) 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" 
    return text.strip()

@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def get_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversational_chain(vectorstore):
    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_length=MAX_TOKENS,
        temperature=0.3, 
        do_sample=False 
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        max_token_limit=800
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}), 
        memory=memory,
        verbose=False
    )

def handle_userinput(user_question):
    if st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                template = user_template if i % 2 == 0 else bot_template
                st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process your PDFs first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📚")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from these PDFs")
                        return
                        
                    text_chunks = get_text_chunks(raw_text)
                    embeddings = load_embedding_model()
                    vectorstore = get_vectorstore(text_chunks, embeddings)
                    st.session_state.conversation = get_conversational_chain(vectorstore)
                    st.success("Ready to answer questions!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()