# 📄 DocStudio - Abstractive Document Summarizer

A powerful, flexible document summarization system that combines local HuggingFace models with optional OpenAI integration for high-quality abstractive summaries.

## ✨ Features

- **🔧 Local Processing**: Use HuggingFace Transformers for offline summarization
- **☁️ Cloud Integration**: Optional OpenAI API for enhanced quality
- **📁 Multi-Format Support**: PDF, DOCX, DOC, and TXT files
- **🎯 Abstractive Summarization**: Generate human-like summaries, not just extracts
- **⚡ Fast API**: RESTful API with FastAPI
- **🎨 Modern UI**: Beautiful web interface with drag-and-drop
- **📊 Rich Analytics**: Document statistics and processing metadata
- **🔧 Configurable**: Customize summarization parameters

## 🏗️ Architecture

```
DocStudio/
├── src/                    # Core Python modules
│   ├── document_processor.py  # Document extraction & cleaning
│   ├── summarizer.py         # Local & cloud summarization
│   └── main.py              # Main application interface
├── api/                    # FastAPI web service
│   └── app.py              # REST API endpoints
├── ui/                     # Web interface
│   └── index.html          # Modern HTML/JS UI
├── test_summarizer.py      # Test script
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd DocStudio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Installation

```bash
# Run the test script
python test_summarizer.py
```

### 3. Start the API Server

```bash
# Start the FastAPI server
cd api
python app.py
```

The API will be available at `http://localhost:8000`

### 4. Use the Web Interface

Open `ui/index.html` in your browser or serve it with a local server:

```bash
# Using Python's built-in server
cd ui
python -m http.server 8080
```

Then visit `http://localhost:8080`

## 📖 Usage Examples

### Python API

```python
from src.main import DocStudio, create_summarization_config

# Initialize DocStudio
doc_studio = DocStudio()

# Summarize text
text = "Your long text here..."
result = doc_studio.summarize_text(text)

print(f"Summary: {result['summary']}")
print(f"Processing time: {result['processing_time_seconds']:.2f}s")

# Process a document
result = doc_studio.process_document("path/to/document.pdf")
print(f"Document summary: {result['summary']}")

# Custom configuration
config = create_summarization_config(
    max_length=300,
    min_length=50,
    temperature=0.8,
    num_beams=3
)

result = doc_studio.summarize_text(text, config=config)
```

### REST API

```bash
# Summarize text
curl -X POST "http://localhost:8000/summarize/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "max_length": 512,
    "min_length": 50,
    "temperature": 1.0,
    "num_beams": 4
  }'

# Upload and summarize document
curl -X POST "http://localhost:8000/summarize/document" \
  -F "file=@document.pdf" \
  -F "max_length=512" \
  -F "use_openai=false"

# Get system information
curl "http://localhost:8000/system/info"
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: OpenAI API key for enhanced summaries
export OPENAI_API_KEY="your-openai-api-key"
```

### Summarization Parameters

- **max_length**: Maximum length of the summary (default: 512)
- **min_length**: Minimum length of the summary (default: 50)
- **temperature**: Sampling temperature for creativity (default: 1.0)
- **num_beams**: Number of beams for beam search (default: 4)
- **use_openai**: Whether to use OpenAI API (default: false)

## 🤖 Models

### Local Models (HuggingFace)

- **Default**: `facebook/bart-large-cnn` - Optimized for summarization
- **Alternative**: `t5-base` - General-purpose text generation
- **Custom**: Any HuggingFace summarization model

### Cloud Models (OpenAI)

- **GPT-3.5-turbo**: Fast and cost-effective
- **GPT-4**: Highest quality (requires API access)

## 📁 Supported Formats

- **PDF**: Full text extraction with metadata
- **DOCX/DOC**: Microsoft Word documents
- **TXT**: Plain text files (multiple encodings)

## 🧪 Testing

```bash
# Run the test suite
python test_summarizer.py

# Test specific functionality
python -c "
from src.main import DocStudio
doc_studio = DocStudio()
result = doc_studio.summarize_text('Test text for summarization.')
print('Success!' if result['success'] else 'Failed')
"
```

## 🔍 API Documentation

When the API server is running, visit:
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

- `GET /` - API information
- `POST /summarize/text` - Summarize text input
- `POST /summarize/document` - Summarize uploaded document
- `GET /system/info` - System information
- `GET /health` - Health check

## 🛠️ Development

### Project Structure

```
src/
├── document_processor.py  # Document handling
├── summarizer.py         # Summarization logic
└── main.py              # Main interface

api/
└── app.py              # FastAPI application

ui/
└── index.html          # Web interface
```

### Adding New Features

1. **New Document Format**: Extend `DocumentProcessor` class
2. **New Model**: Add to `LocalSummarizer` or `OpenAISummarizer`
3. **New API Endpoint**: Add to `api/app.py`
4. **UI Enhancement**: Modify `ui/index.html`

## 📊 Performance

### Benchmarks

- **Local Model**: ~2-5 seconds for 1000 words
- **OpenAI API**: ~1-3 seconds for 1000 words
- **Document Processing**: ~1-2 seconds for typical documents

### Memory Usage

- **Local Model**: ~2-4 GB RAM
- **Document Processing**: ~100-500 MB RAM
- **API Server**: ~1-2 GB RAM

## 🔒 Security Considerations

- **File Upload**: Validate file types and sizes
- **API Keys**: Store securely in environment variables
- **CORS**: Configure appropriately for production
- **Rate Limiting**: Implement for production use

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Setup

1. **Environment**: Use production-grade server (Gunicorn + Nginx)
2. **Security**: Implement authentication and rate limiting
3. **Monitoring**: Add logging and health checks
4. **Scaling**: Use load balancers for high traffic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for the AI community** 
