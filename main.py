import os
import socket
from typing import List, Dict, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer
from groq import Groq
from PyPDF2 import PdfReader
import base64
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with all necessary clients and configurations"""
        # Initialize Qdrant client with increased timeout
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=300  # Increased timeout to 5 minutes
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Collection name in Qdrant
        self.collection_name = "pdf_documents"
        
        # Batch size for processing
        self.batch_size = 10
        
        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create a collection in Qdrant if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2  # Reduce segment number for better performance
                    )
                )
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def store_pdf(self, pdf_path: str) -> Optional[str]:
        """Store PDF file content as base64 string with compression"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_bytes = file.read()
                # Only store the first 1MB of the PDF to avoid size issues
                pdf_bytes = pdf_bytes[:1024*1024]  
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                return pdf_base64
        except Exception as e:
            print(f"Error storing PDF: {e}")
            return None

    def process_pdf(self, pdf_path: str, chunk_size: int = 500) -> List[Dict]:
        """Process PDF file and split into smaller chunks"""
        try:
            pdf_reader = PdfReader(pdf_path)
            chunks = []
            current_chunk = []
            current_size = 0
            current_page = 1
            
            # Store PDF content once
            pdf_base64 = self.store_pdf(pdf_path)
            
            print("Processing PDF pages...")
            for page_num, page in enumerate(tqdm(pdf_reader.pages), 1):
                text = page.extract_text()
                # Split by sentences approximately
                sentences = text.replace('. ', '.|').split('|')
                
                for sentence in sentences:
                    if len(sentence.strip()) == 0:
                        continue
                        
                    current_chunk.append(sentence)
                    current_size += len(sentence)
                    
                    if current_size >= chunk_size:
                        chunk_text = " ".join(current_chunk)
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "page": current_page,
                                "pdf_name": os.path.basename(pdf_path),
                                "pdf_content": pdf_base64
                            }
                        })
                        current_chunk = []
                        current_size = 0
                
                current_page = page_num + 1
            
            # Add any remaining text
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "page": current_page - 1,
                        "pdf_name": os.path.basename(pdf_path),
                        "pdf_content": pdf_base64
                    }
                })
            
            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def safe_upload_batch(self, points: List[models.PointStruct], attempt: int = 0):
        """Safely upload a batch of points with retry logic"""
        max_attempts = 3
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except ResponseHandlingException as e:
            if attempt < max_attempts:
                print(f"Upload failed, retrying... (Attempt {attempt + 1})")
                time.sleep(2 ** attempt)  # Exponential backoff
                return self.safe_upload_batch(points, attempt + 1)
            else:
                print(f"Failed to upload batch after {max_attempts} attempts")
                raise

    def index_documents(self, document_chunks: List[Dict]):
        """Index documents in Qdrant Cloud with batch processing"""
        print(f"Indexing {len(document_chunks)} chunks...")
        
        # Process in smaller batches
        batch_size = self.batch_size
        total_batches = (len(document_chunks) + batch_size - 1) // batch_size
        
        for batch_num in tqdm(range(total_batches)):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(document_chunks))
            batch_chunks = document_chunks[start_idx:end_idx]
            
            points = []
            for i, doc in enumerate(batch_chunks):
                try:
                    # Generate embedding
                    embedding = self.encoder.encode(doc["text"])
                    
                    # Create point
                    point = models.PointStruct(
                        id=start_idx + i,
                        vector=embedding.tolist(),
                        payload={
                            "text": doc["text"],
                            "page": doc["metadata"]["page"],
                            "pdf_name": doc["metadata"]["pdf_name"]
                            # Exclude pdf_content from payload to reduce size
                        }
                    )
                    points.append(point)
                except Exception as e:
                    print(f"Error processing chunk {start_idx + i}: {e}")
                    continue
            
            if points:
                try:
                    self.safe_upload_batch(points)
                    time.sleep(0.5)  # Add small delay between batches
                except Exception as e:
                    print(f"Error uploading batch {batch_num + 1}: {e}")
                    continue

    def _check_connection(self):
        """Check internet connectivity before making requests"""
        try:
            # Try to connect to a reliable host
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for relevant documents using semantic search
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata
        """

        try:
            query_vector = self.encoder.encode(query)
            
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [{
                "text": hit.payload["text"],
                "score": hit.score,
                "page": hit.payload["page"],
                "pdf_name": hit.payload["pdf_name"]
            } for hit in search_result]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """
        Generate an answer using the LLM based on the provided context
        
        Args:
            query (str): The user's question
            context (List[Dict]): Relevant document chunks for context
            
        Returns:
            str: Generated answer
        """

        if not context:
            return "Unable to find relevant context to answer the question."
            
        try:
            context_str = "\n".join([
                f"Context {i+1} (Page {doc['page']} of {doc['pdf_name']}):\n{doc['text']}"
                for i, doc in enumerate(context)
            ])
            
            # Construct a detailed prompt with specific instructions
            prompt = f"""You are a helpful AI assistant tasked with answering questions about a document.
            Please analyze the following context carefully and provide a precise, well-structured answer.
            
            CONTEXT:
            {context_str}
            
            QUESTION:
            {query}
            
            INSTRUCTIONS:
            1. Base your answer solely on the provided context
            2. If the context doesn't contain enough information, say so
            3. Be concise but thorough
            4. If summarizing, focus on key points and maintain accuracy
            5. Include relevant page numbers in your answer when appropriate.
            6. If you are asked about to show information in table format, please do so.
            
            ANSWER:"""


            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."

    def query(self, question: str) -> Dict:
        """
        Main method to process queries and generate answers
        
        Args:
            question (str): The user's question
            
        Returns:
            Dict: Answer and source information
        """
        try:

            if not self._check_connection():
                return {
                    "answer": """Unable to process your query due to network connectivity issues. 
                    Please check your internet connection and try again.""",
                    "sources": []
                }

            relevant_docs = self.search(question)
            answer = self.generate_answer(question, relevant_docs)

            relevant_docs = self.search(question)
            answer = self.generate_answer(question, relevant_docs)
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "page": doc["page"],
                        "pdf_name": doc["pdf_name"],
                        "relevance_score": doc["score"]
                    }
                    for doc in relevant_docs
                ]
            }
        except Exception as e:
            error_message = f"""An error occurred while processing your query: {str(e)}
            Please try again!!!"""
            return {
                "answer": error_message,
                "sources": []
            }

def main():
    """Main function to run the RAG system"""
    try:
        # Initialize the RAG system
        rag_system = RAGSystem()
        
        # Example usage: Index a PDF
        pdf_path = "microsoft-annual-report.pdf"
        if os.path.exists(pdf_path):
            print("Processing PDF...")


            chunks = rag_system.process_pdf(pdf_path)
            print(f"Successfully generated {len(chunks)} text chunks from the PDF")
            
            print("Indexing chunks...")

            rag_system.index_documents(chunks)
            print("Indexing complete")
        else:
            print(f"PDF file not found: {pdf_path}")
            return
        
        # Interactive query loop

        print("""\nRAG System Ready!
        You can now ask questions about the document.
        Type 'quit' to exit the program.""")

        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            print("\nProcessing query...")
            result = rag_system.query(question)
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print("\nSources:")
                for source in result['sources']:
                    print(f"- Page {source['page']} of {source['pdf_name']} (Score: {source['relevance_score']:.2f})")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()