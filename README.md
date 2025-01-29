# README.md
# RAG System with Multi-Format Support

A Retrieval-Augmented Generation (RAG) system that supports multiple document formats including PDF.

## Features
- Semantic search using Qdrant
- LLM integration with Groq
- Batch processing with error handling
- Network connectivity checks
- PDF support with PyPDF2

## API Setup Instruction
# 1. Get Groq API Key:
Go to console.groq.com
- Sign up & verify email
- Go to API Keys section
- Create new key & copy it
- Save as GROQ_API_KEY in .env file

# 2. Get Qdrant API Key:
Go to cloud.qdrant.io
- Sign up & verify email
- Create a new cluster (choose free tier)
- Wait 2-3 minutes for initialization
- Copy API Key and URL from Credentials section
  - **API Key**: Click "Create API Key"
  - **URL**: Copy the cluster URL (looks like `https://xxx-xxx-xxx.aws.cloud.qdrant.io`)
- Save as QDRANT_API_KEY and QDRANT_URL in .env file


## Requirements
- Python 3.8+
- Qdrant Cloud account
- Groq API access


## Environment Variables
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your API keys


## Usage
```python
python main.py
```
