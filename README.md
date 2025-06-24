# RAG Chatbot with PDF

A powerful RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- 📄 PDF document upload and processing
- 🤖 AI-powered question answering using RAG
- 💬 Chat-like interface for easy interaction
- 🔍 Intelligent document retrieval and context understanding
- 🌐 Web-based interface accessible from anywhere

## Live Demo

[Deploy to Streamlit Cloud](https://share.streamlit.io/)

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select your repository
5. Deploy!

### Option 2: Other Platforms

- **Railway**: Easy deployment with automatic scaling
- **Render**: Free tier available
- **Heroku**: Requires additional configuration

## Environment Variables

Make sure to set your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Technologies Used

- Streamlit - Web framework
- LangChain - RAG framework
- FAISS - Vector database
- HuggingFace - Embeddings
- OpenAI/OpenRouter - LLM provider 