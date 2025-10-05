# PDF RAG Chatbot with Google Gemini

A precise RAG (Retrieval-Augmented Generation) chatbot for extracting exact information from PDF documents using Google Gemini, LangChain, and Streamlit.

## Features

- **Zero Randomness**: Temperature set to 0 for deterministic, exact answers
- **PDF Parsing**: Handles 50-70 page PDFs efficiently
- **Source Citations**: Shows exact excerpts from the PDF used to generate answers
- **Precise Retrieval**: Uses FAISS vector search for accurate context retrieval
- **Clean UI**: Simple Streamlit interface

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Local Development

1. Run the app:

```bash
streamlit run app.py
```

2. Enter your Google API key in the sidebar
3. Upload your PDF (50-70 pages)
4. Click "Process PDF"
5. Ask questions and get exact answers from the document

### Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app
4. In app settings, add your secrets:
   - Go to "Settings" → "Secrets"
   - Add: `GOOGLE_API_KEY = "your_api_key_here"`
5. Your app will automatically use the secret API key

**Note**: Max upload size is set to 200MB in config.toml

## How It Works

1. **PDF Extraction**: Extracts all text from the uploaded PDF
2. **Text Chunking**: Splits text into 1000-character chunks with 200-character overlap
3. **Embeddings**: Creates vector embeddings using Google's embedding-001 model
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds the 4 most relevant chunks for each question
6. **Generation**: Gemini-pro generates answers with temperature=0 (no randomness)

## Configuration

Key settings for precision:

- `temperature=0`: Eliminates randomness in responses
- `chunk_size=1000`: Optimal for 50-70 page documents
- `chunk_overlap=200`: Preserves context across chunks
- `k=4`: Retrieves top 4 relevant chunks per query

## Notes

- Answers are strictly based on PDF content
- If information isn't in the PDF, the bot will say so
- Source excerpts are shown for verification
