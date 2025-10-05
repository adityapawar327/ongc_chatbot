import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_vectorstore(text, api_key):
    """Create FAISS vectorstore from text"""
    # Split text into chunks with overlap for context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using free HuggingFace model (no quota limits)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def create_qa_chain(vectorstore, api_key):
    """Create QA chain with precise retrieval settings"""
    # Custom prompt for exact answers with clear formatting
    prompt_template = """You are a helpful assistant that answers questions based on PDF documents.

Use the following context from the PDF to answer the question accurately and concisely.

Rules:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I cannot find this information in the PDF"
- Provide specific information and quotes from the PDF
- Format your answer clearly with bullet points or paragraphs as appropriate
- Do not add external knowledge

Context from PDF:
{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Initialize Gemini with temperature=0 for deterministic responses
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0,  # No randomness
        convert_system_message_to_human=True
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# UI
st.title("📚 PDF RAG Chatbot with Google Gemini")
st.markdown("Upload a PDF and ask questions - get exact answers from the document")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Try to get API key from multiple sources
    api_key = None
    
    # 1. Try .env file
    if os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.success("✓ API Key loaded from .env file")
    # 2. Try Streamlit secrets (for cloud deployment)
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✓ API Key loaded from secrets")
    # 3. Manual input
    else:
        api_key = st.text_input("Google API Key", type="password", help="Enter your Google Gemini API key")
        st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file and api_key:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Extract text
                    text = extract_pdf_text(uploaded_file)
                    st.success(f"Extracted {len(text)} characters")
                    
                    # Create vectorstore
                    st.session_state.vectorstore = create_vectorstore(text, api_key)
                    
                    # Create QA chain
                    st.session_state.conversation = create_qa_chain(
                        st.session_state.vectorstore, 
                        api_key
                    )
                    
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main chat interface
if st.session_state.conversation:
    st.subheader("Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your PDF"):
        # Display user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF..."):
                try:
                    response = st.session_state.conversation({"query": question})
                    
                    # Extract the answer text
                    if isinstance(response, dict) and "result" in response:
                        answer = response["result"]
                    else:
                        answer = str(response)
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Show source documents if available
                    if isinstance(response, dict) and "source_documents" in response:
                        with st.expander("📄 View Source Excerpts"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Excerpt {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                                if i < len(response["source_documents"]) - 1:
                                    st.divider()
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF and enter your API key to start chatting")
