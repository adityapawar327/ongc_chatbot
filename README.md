# PDF RAG Chatbot with Google Gemini

A precise RAG (Retrieval-Augmented Generation) chatbot for extracting exact information from PDF documents using Google Gemini 2.0 Flash, LangChain, and Streamlit.

## Features

- **Zero Randomness**: Temperature set to 0 for deterministic, exact answers
- **PDF Parsing**: Handles 50-70 page PDFs efficiently
- **Free Embeddings**: Uses HuggingFace sentence-transformers (no API quota limits)
- **Source Citations**: Shows exact excerpts from the PDF used to generate answers
- **Precise Retrieval**: Uses FAISS vector search for accurate context retrieval
- **Clean UI**: Simple Streamlit interface
- **Cloud Ready**: Works both locally and on Streamlit Cloud

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key (free tier available)

## Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/adityapawar327/ongc_chatbot.git
cd ongc_chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- langchain & langchain-google-genai
- sentence-transformers (for free embeddings)
- faiss-cpu (for vector storage)
- PyPDF2 (for PDF parsing)
- python-dotenv (for environment variables)

### 4. Get Your Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 5. Configure API Key

Create a `.env` file in the project root:

```bash
# Windows
echo GOOGLE_API_KEY=your_api_key_here > .env

# macOS/Linux
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Or manually create `.env` and add:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload PDF**: Click "Upload PDF" in the sidebar and select your PDF file (50-70 pages recommended)
2. **Process PDF**: Click "Process PDF" button - this will:
   - Extract text from the PDF
   - Create embeddings using HuggingFace (runs locally, no API calls)
   - Store vectors in FAISS for fast retrieval
3. **Ask Questions**: Type your questions in the chat input
4. **View Answers**: Get precise answers with source excerpts from the PDF

## Streamlit Cloud Deployment

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `adityapawar327/ongc_chatbot`
4. Branch: `main`
5. Main file path: `app.py`
6. Click "Advanced settings"
7. In "Secrets" section, add:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```
8. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

## How It Works

1. **PDF Extraction**: Extracts all text from the uploaded PDF using PyPDF2
2. **Text Chunking**: Splits text into 1000-character chunks with 200-character overlap
3. **Embeddings**: Creates vector embeddings using HuggingFace's `all-MiniLM-L6-v2` model (free, runs locally)
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds the 4 most relevant chunks for each question
6. **Generation**: Gemini 2.0 Flash generates answers with temperature=0 (no randomness)

## Configuration

Key settings for precision:
- **Model**: `gemini-2.0-flash-exp` (latest Gemini model)
- **Temperature**: `0` (eliminates randomness in responses)
- **Chunk Size**: `1000` characters (optimal for 50-70 page documents)
- **Chunk Overlap**: `200` characters (preserves context across chunks)
- **Retrieval**: Top `4` relevant chunks per query
- **Embeddings**: Free HuggingFace model (no quota limits)

## Project Structure

```
ongc_chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # API key (local only, not in git)
├── .env.example          # Example environment file
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Troubleshooting

### "Could not import sentence_transformers"
```bash
pip install sentence-transformers --upgrade
```

### "API key not found"
- Make sure `.env` file exists with `GOOGLE_API_KEY=your_key`
- Or enter API key manually in the sidebar

### "Quota exceeded" error
- The app uses free HuggingFace embeddings (no quota)
- Only Gemini API calls count toward quota (15 requests/minute free tier)
- Wait a minute and try again

### Slow first run
- First time loading the embedding model takes 1-2 minutes
- Model is cached for subsequent runs

## Notes

- Answers are strictly based on PDF content
- If information isn't in the PDF, the bot will say so
- Source excerpts are shown for verification
- Embeddings run locally (no external API calls)
- Only chat responses use Gemini API

## License

MIT License

## Author

Aditya Pawar
