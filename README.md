# 📚 Production-Grade PDF RAG Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) chatbot for analyzing PDF documents using **Google Gemini 2.0 Flash**, **LangChain**, and **Streamlit**.

## ✨ Features

- 🔍 **Comprehensive Search**: Searches entire PDF with advanced MMR retrieval
- 🎯 **High Accuracy**: Optimized chunking and retrieval for precise answers
- 📄 **Source Citations**: Shows exact excerpts used for each answer
- 🚀 **Production-Grade**: Built for reliability and performance
- 💰 **100% Free**: Uses free HuggingFace embeddings + Gemini free tier
- ☁️ **Cloud-Ready**: Deploy to Streamlit Cloud in minutes

## 🚀 Quick Start

### 1. Get API Key

Get your free Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

## 🌐 Deploy to Streamlit Cloud

1. **Push to GitHub** (don't include `.env` file)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. In "Advanced settings" → "Secrets", add:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```
5. Click "Deploy"

**See `SETUP_INSTRUCTIONS.md` for detailed deployment guide**

## 🛠️ How It Works

1. **PDF Extraction**: Extracts text with page tracking
2. **Smart Chunking**: 800-character chunks with 150-char overlap
3. **Free Embeddings**: HuggingFace sentence-transformers (runs locally)
4. **Vector Search**: FAISS with MMR for diverse, relevant results
5. **AI Generation**: Gemini 2.0 Flash synthesizes comprehensive answers

## ⚙️ Advanced Configuration

### Retrieval Settings

- **Chunks retrieved**: 8 (comprehensive coverage)
- **Candidates considered**: 20 (MMR diversity)
- **Search type**: MMR (Maximum Marginal Relevance)
- **Lambda**: 0.7 (balance relevance vs diversity)

### AI Settings

- **Model**: gemini-2.0-flash-exp
- **Temperature**: 0.1 (slight creativity for synthesis)
- **Max tokens**: 2048 (longer, detailed responses)

### Chunking Strategy

- **Chunk size**: 800 characters
- **Overlap**: 150 characters
- **Separators**: Smart splitting on paragraphs, sentences, words

## 📊 Performance

- ✅ Handles PDFs up to 200MB
- ✅ Processes 50-70 page documents in ~30 seconds
- ✅ Free tier: 15 requests/minute (Gemini API)
- ✅ Unlimited embeddings (runs locally)

## 🔒 Security

- API keys stored in `.env` (local) or Secrets (cloud)
- `.gitignore` prevents accidental key exposure
- No data stored or logged

## 📝 Usage Tips

- Ask specific questions for best results
- Try questions about dates, numbers, or specific topics
- Review source excerpts to verify answers
- For scanned PDFs, use OCR preprocessing first

## 🤝 Sharing with Others

To share this project:

1. Create a ZIP file of the project folder
2. Include `SETUP_INSTRUCTIONS.md` and `QUICK_START.txt`
3. Remove your `.env` file before sharing
4. Recipients follow instructions in `SETUP_INSTRUCTIONS.md`

## 📄 License

Free to use and modify for personal and commercial projects.

## 🙏 Credits

Built with:

- [Google Gemini](https://ai.google.dev/) - AI model
- [LangChain](https://langchain.com/) - RAG framework
- [Streamlit](https://streamlit.io/) - Web interface
- [HuggingFace](https://huggingface.co/) - Free embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
