# Knowledge Base Chatbot

A free, open-source RAG (Retrieval-Augmented Generation) chatbot that uses documents from your Google Drive as knowledge base.

## Features

- 📄 **Document Support**: PDFs, DOCX, Google Docs, and TXT files
- 🔍 **Smart Search**: Vector-based document retrieval using sentence embeddings
- 🤖 **AI-Powered**: Uses Google Gemini 1.5 Flash (free tier) for intelligent responses
- 🌐 **Web Interface**: Built with Streamlit for easy deployment
- 💾 **Local Storage**: ChromaDB for persistent vector storage
- 🔐 **Secure**: OAuth-based Google Drive authentication

## Architecture

```
Google Drive (your folder)
       │
       ▼  Drive API
Download & Extract Text (PDFs, DOCX, Google Docs, TXT)
       │
       ▼  sentence-transformers (CPU, free)
Chunk → Embed → ChromaDB (local vector store)
       │
       ▼
User question → embed → retrieve relevant chunks → LLM → answer
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
cd ~/kb-chatbot

# Activate virtual environment
source .venv/Scripts/activate

# Run setup
python setup.py
```

### 2. Get API Keys

- **Google AI API Key**: Get your free key from https://makersuite.google.com/app/apikey
- **Google Drive**: Already configured via OAuth setup

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Configure in Web Interface

1. Enter your Google AI API key in the sidebar
2. Click "Setup Google Drive" to authenticate
3. Click "Process Documents" to index your files
4. Start chatting!

## Detailed Setup

### Prerequisites

- Python 3.11+
- Google account (hetalinpatelin@gmail.com)
- Google Cloud project with Drive API enabled

### Step 1: Google Cloud Setup

1. Go to https://console.cloud.google.com/projectselector2/home/dashboard
2. Create a new project (name: "KB-Chatbot")
3. Enable these APIs:
   - Google Drive API
   - Google Docs API
4. Create OAuth 2.0 Client ID (Desktop app)
5. Add yourself as a test user at https://console.cloud.google.com/auth/audience
6. Download the JSON file and provide the path when prompted

### Step 2: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or run setup script
python setup.py
```

### Step 3: Run and Configure

1. Start the app: `streamlit run app.py`
2. Open http://localhost:8501 in your browser
3. Enter your Google AI API key
4. Click "Setup Google Drive"
5. Click "Process Documents" to index your Google Drive files

## Usage

### Processing Documents

- The app will automatically find and process documents in your Google Drive
- Supported formats: PDF, DOCX, Google Docs, TXT
- Documents are chunked into 1000-character segments with 200-character overlap
- Each chunk is embedded and stored in ChromaDB

### Chatting

- Type questions about your documents in the chat interface
- The app searches for relevant document chunks
- Google Gemini generates answers based on the retrieved context
- Responses include source document information

### Collection Management

- **Show Collection Stats**: View indexed files and statistics
- **Clear Collection**: Remove all documents (start fresh)

## Deployment

### Local Development

For development and testing:
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Deploy with `streamlit run app.py`

### Hugging Face Spaces (Free)

1. Push your code to GitHub
2. Go to https://huggingface.co/spaces
3. Create a new space
4. Use the Streamlit template

### Self-Hosted (Paid)

For production deployment on a VPS:
```bash
# Install dependencies
pip install -r requirements.txt

# Run with systemd service
sudo systemctl enable kb-chatbot
sudo systemctl start kb-chatbot
```

## Cost Breakdown

### Free Tier (Recommended for Start)
- **Google Gemini 1.5 Flash**: 60 requests/minute, 1,500 requests/day
- **Streamlit Cloud**: Free deployment with limitations
- **ChromaDB**: Free local storage
- **Sentence Transformers**: Free CPU-based embeddings

### Paid Options (When Needed)
- **Google Gemini Pro**: ~$0.000275/1K tokens
- **VPS Hosting**: $5-20/month for better performance
- **GPU Acceleration**: For large document collections

## File Structure

```
kb-chatbot/
├── app.py                 # Main Streamlit application
├── document_processor.py # Document extraction and processing
├── rag_pipeline.py       # RAG pipeline with ChromaDB
├── setup.py              # Setup and testing script
├── requirements.txt      # Python dependencies
├── chroma_db/            # Vector database storage
├── downloads/            # Temporary file downloads
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **Google Drive Authentication Failed**
   - Ensure you added yourself as a test user
   - Check the OAuth client ID in Google Cloud Console
   - Verify the JSON file path is correct

2. **API Key Issues**
   - Get a new key from https://makersuite.google.com/app/apikey
   - Ensure the key has "Generative Language API" enabled

3. **Model Loading Errors**
   - Check internet connection for model downloads
   - Ensure you have enough disk space (~100MB for embeddings model)

4. **Document Processing Errors**
   - Verify documents are in supported formats
   - Check file permissions and access

### Performance Tips

- For large document collections (>1000 documents), consider GPU acceleration
- Use sentence boundaries for better chunking
- Monitor API usage to stay within free tier limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use and modify for your own projects.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the setup steps
3. Ensure all dependencies are installed correctly