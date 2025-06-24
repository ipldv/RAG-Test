import os
os.environ["OPENAI_API_KEY"] = "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"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile

st.set_page_config(
    page_title="RAG Chatbot with PDF",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Rayan's RAG Chatbot for Your PDF")
st.header("1. Upload Your PDF")

# Session state cleanup
# If a file is already processed, but we're back on the main page,
# offer to clear it.
if "qa_chain" in st.session_state:
    st.info(f"You have already processed `{st.session_state.uploaded_filename}`.")
    if st.button("Upload a New PDF"):
        del st.session_state.qa_chain
        del st.session_state.uploaded_filename
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.rerun()

else:
    # رفع ملف PDF
    uploaded_file = st.file_uploader("Upload a PDF to get started", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing your PDF..."):
            # Save file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(pages)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings()
            vectordb = FAISS.from_documents(docs, embeddings)

            # Build RAG chain
            retriever = vectordb.as_retriever()
            llm = ChatOpenAI(
                model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            # Store chain and file name in session state
            st.session_state.qa_chain = qa_chain
            st.session_state.uploaded_filename = uploaded_file.name
        
        st.success(f"Successfully processed `{uploaded_file.name}`.")
        st.header("2. Ask Questions")
        st.info("Navigate to the **Chat** page in the sidebar to start asking questions about your document.")

# --- ChatGPT-like chat UI CSS ---
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        color: #000;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 8px;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        color: #000;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 8px;
        max-width: 70%;
        margin-right: auto;
        margin-left: 0;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Check if the QA chain is in session state
if "qa_chain" not in st.session_state:
    st.warning("Please go to the main page and upload a PDF first.")
    st.stop()

st.title("💬 Chat with your PDF")
st.info(f"Ready to answer questions about **{st.session_state.uploaded_filename}**.")

# Initialize chat history in session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages from session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input.
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user's message to session state and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response.
    with st.chat_message("assistant"):
        with st.spinner():
            # Call the QA chain to get the result.
            response = st.session_state.qa_chain.run(prompt)
            st.markdown(response)

    # Add assistant's response to session state.
    st.session_state.messages.append({"role": "assistant", "content": response})