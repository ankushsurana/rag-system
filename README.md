# README.md
# RAG System with Multi-Format Support

A Retrieval-Augmented Generation (RAG) system that supports multiple document formats including PDF.

## Features
- Semantic search using Qdrant
- LLM integration with Groq
- Batch processing with error handling
- Network connectivity checks
- PDF support with PyPDF2

## Requirements
- Python 3.8+
- Qdrant Cloud account
- Groq API access


## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your API keys


## Environment Variables
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage
```python
python main.py
```