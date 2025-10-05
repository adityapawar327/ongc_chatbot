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
if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = None

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF with page tracking"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    page_count = len(pdf_reader.pages)
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        page_text = page.extract_text()
        if page_text.strip():  # Only add non-empty pages
            text += f"\n--- Page {page_num} ---\n{page_text}"
    
    return text, page_count

def create_vectorstore(text, api_key):
    """Create FAISS vectorstore from text with optimized chunking"""
    # Optimized chunking for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better precision
        chunk_overlap=150,  # Good overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using free HuggingFace model (no quota limits)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vectorstore with metadata
    metadatas = [{"chunk_id": i, "total_chunks": len(chunks)} for i in range(len(chunks))]
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    return vectorstore

def create_qa_chain(vectorstore, api_key):
    """Create production-grade QA chain with comprehensive retrieval"""
    # Enhanced prompt for production-grade responses
    prompt_template = """You are an expert document analysis assistant. Your task is to provide accurate, comprehensive answers based on the PDF document.

INSTRUCTIONS:
1. Analyze ALL the provided context carefully
2. Extract and synthesize relevant information from multiple sections if needed
3. Provide complete, well-structured answers
4. Use bullet points for lists, clear paragraphs for explanations
5. Include specific details, numbers, dates, and quotes when available
6. If information is partial, state what you found and what's missing
7. Only say "I cannot find this information in the PDF" if truly absent

CONTEXT FROM PDF:
{context}

QUESTION: {question}

COMPREHENSIVE ANSWER:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Initialize Gemini with optimized settings
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.1,  # Slight creativity for better synthesis
        max_output_tokens=2048,  # Allow longer responses
        convert_system_message_to_human=True
    )
    
    # Create retrieval chain with enhanced retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 8,  # Retrieve more chunks for comprehensive coverage
                "fetch_k": 20,  # Consider more candidates
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# UI
st.title("📚 Production-Grade PDF RAG Chatbot")
st.markdown("**Powered by Google Gemini 2.0 Flash** | Comprehensive document analysis with advanced retrieval")

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
        if st.button("Process PDF", type="primary"):
            with st.spinner("🔄 Processing PDF..."):
                try:
                    # Extract text with page info
                    text, page_count = extract_pdf_text(uploaded_file)
                    
                    if not text.strip():
                        st.error("❌ Could not extract text from PDF. Please ensure it's not a scanned image.")
                        st.stop()
                    
                    st.info(f"📄 Extracted {len(text):,} characters from {page_count} pages")
                    
                    # Create vectorstore
                    with st.spinner("🧠 Creating embeddings..."):
                        st.session_state.vectorstore = create_vectorstore(text, api_key)
                    
                    # Create QA chain
                    with st.spinner("⚙️ Initializing AI model..."):
                        st.session_state.conversation = create_qa_chain(
                            st.session_state.vectorstore, 
                            api_key
                        )
                    
                    st.session_state.pdf_info = {
                        "name": uploaded_file.name,
                        "pages": page_count,
                        "chars": len(text)
                    }
                    
                    st.success("✅ PDF processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# Main chat interface
if st.session_state.conversation:
    # Show PDF info
    if st.session_state.pdf_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", st.session_state.pdf_info["name"])
        with col2:
            st.metric("📑 Pages", st.session_state.pdf_info["pages"])
        with col3:
            st.metric("💬 Questions Asked", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
    
    st.divider()
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask anything about your PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing document..."):
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
                    if isinstance(response, dict) and "source_documents" in response and len(response["source_documents"]) > 0:
                        with st.expander(f"📚 View {len(response['source_documents'])} Source Excerpts"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                # Show more context
                                content = doc.page_content
                                if len(content) > 500:
                                    st.text(content[:500] + "...")
                                else:
                                    st.text(content)
                                
                                # Show metadata if available
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Chunk {doc.metadata.get('chunk_id', 'N/A')} of {doc.metadata.get('total_chunks', 'N/A')}")
                                
                                if i < len(response["source_documents"]):
                                    st.divider()
                except Exception as e:
                    answer = f"❌ Error processing question: {str(e)}"
                    st.error(answer)
                    import traceback
                    with st.expander("🔧 Debug Info"):
                        st.code(traceback.format_exc())
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    # Welcome screen
    st.info("👈 **Get Started:** Upload a PDF and click 'Process PDF' to begin")
    
    st.markdown("""
    ### 🚀 Features:
    - **Comprehensive Search**: Searches entire PDF with advanced retrieval
    - **Smart Chunking**: Optimized text splitting for better accuracy
    - **Context-Aware**: Retrieves multiple relevant sections
    - **Source Citations**: Shows exact excerpts used for answers
    - **Production-Grade**: Built for reliability and performance
    
    ### 💡 Tips:
    - Ask specific questions for best results
    - Try questions about dates, numbers, or specific topics
    - Review source excerpts to verify answers
    """)
