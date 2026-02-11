import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Vector DB configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required API key
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set.")
    print("Please set it in your .env file or environment variables.")

class RedditRAG:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store if it exists
        try:
            self.vector_store = FAISS.load_local(
                VECTOR_DB_PATH,
                self.embeddings
            )
            print(f"Loaded vector store from {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Create empty vector store if it doesn't exist
            self.vector_store = FAISS.from_documents(
                [Document(page_content="Initialization document", metadata={"source": "init"})],
                self.embeddings
            )
            self.vector_store.save_local(VECTOR_DB_PATH)
        
        # Initialize LLM
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model="llama-3.1-70b-versatile",
                temperature=0.3,
                max_tokens=2048
            )
            print("Successfully initialized Groq LLM")
        except Exception as e:
            print(f"Error initializing Groq LLM: {e}")
            print("This may be due to an invalid API key or connection issues.")
            # Continue without raising to allow the application to start,
            # but it will fail when trying to use the LLM
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about market-related news.
        
        Use the following news headlines and summaries to answer the user's question. If the news doesn't contain
        relevant information to answer the question, say so and provide general market information.
        
        For each article you reference, include the news headline and a brief summary.
        
        News Articles:
        {context}
        
        User Question: {question}
        
        Important: Always provide a detailed and informative answer based on the provided news articles.
        If you don't have enough information, clearly state that and provide general knowledge about the topic.
        """)
        
        # Create document chain
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # Create retrieval chain
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
    
    def query(self, question, lookback_hours=24):
        """Query the RAG system with a user question"""
        # Convert lookback_hours to timestamp for filtering
        # In a real implementation, we would filter by timestamp
        # For simplicity, we'll just use the retriever as is
        
        try:
            # Try with the question parameter (common in newer versions)
            print("Attempting retrieval_chain.invoke({\"question\": question})")
            response = self.retrieval_chain.invoke({"question": question})
        except Exception as e1:
            print(f"First attempt failed: {str(e1)}")
            try:
                # Try with input parameter (used in some versions)
                print("Attempting retrieval_chain.invoke({\"input\": question})")
                response = self.retrieval_chain.invoke({"input": question})
            except Exception as e2:
                print(f"Second attempt failed: {str(e2)}")
                # Check if the chain has a run method (used in older versions)
                if hasattr(self.retrieval_chain, "run"):
                    print("Attempting retrieval_chain.run(question)")
                    response = {"answer": self.retrieval_chain.run(question)}
                else:
                    # If all else fails, recreate the chain with a different structure
                    print("Recreating chain with different structure")
                    from langchain.chains import RetrievalQA
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=self.retriever,
                        return_source_documents=True
                    )
                    response = qa_chain({"query": question})
        
        # Format response
        result = {
            "answer": response.get("result", "No answer found"),
            "sources": []
        }
        
        # Add sources if available - handle different response structures
        source_docs = None
        
        # Try different possible locations for source documents
        if "context" in response and hasattr(response["context"], "documents"):
            source_docs = response["context"].documents
        elif "source_documents" in response:
            source_docs = response["source_documents"]
        elif "context" in response and isinstance(response["context"], list):
            source_docs = response["context"]
        
        if source_docs:
            for doc in source_docs:
                # Check if it's a Document object
                if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    # Extract source information based on metadata source type
                    if "source" in doc.metadata:                        
                        if doc.metadata["source"] == "yahoo_finance":
                            # Extract information from Yahoo Finance sources
                            # The page_content should contain headline, summary, and URL
                            content = doc.page_content
                            
                            # Try to extract title and URL if possible
                            title = content
                            url = ""
                            
                            # If the content contains a URL, extract it
                            if "http" in content:
                                parts = content.split("http")
                                if len(parts) > 1:
                                    title = parts[0].strip()
                                    url = "http" + parts[1].strip()
                            
                            source = {
                                "title": title,
                                "url": url,
                                "type": "news",
                                "source": "Yahoo Finance"
                            }
                            result["sources"].append(source)
                    else:
                        # Generic handling for documents without a specific source type
                        source = {
                            "title": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        result["sources"].append(source)
        
        return result


# Singleton instance
_reddit_rag = None

def get_reddit_rag():
    """Get singleton instance of RedditRAG"""
    global _reddit_rag
    if _reddit_rag is None:
        _reddit_rag = RedditRAG()
    return _reddit_rag
