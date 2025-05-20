import subprocess
import sys
import os
print("\nManaging dependencies...\nThis might take a few seconds...\n")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'requirements.txt')}"], stdout = subprocess.DEVNULL)

from document_loader import DocumentLoader
from embeddings import VectorStore
from retrieval import Retriever
from llm import LLMService
from agent import Agent

class main:
    def __init__(self):
        """Set up the RAG agent system."""
        
        print("Setting up RAG agent system...")
        
        self.loader = DocumentLoader()
        self.vector_store = VectorStore()
        
        self.documents = self.loader.load_documents()
        self.chunks = self.loader.chunk_documents(self.documents)
        self.vector_store.add_chunks(self.chunks)
        self.llm_service = LLMService()
        
        self.retriever = Retriever(self.vector_store)
        
        self.agent = Agent(self.retriever, self.llm_service)
        
        self.logs = None

    def cli_interface(self) -> None:
        """Run a simple CLI interface for the RAG agent.
        
        Returns:
            None
        """
        
        print("\n\nRAG Agent System ready.\nType 'exit' to quit or 'logs' to see the logs of the last query.")
        print("Example query:")
        print("  - What is your flagship product?")
        
        while True:
            
            query = input("Hi! How can I help you?\n")
            
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            
            if query.lower() in ["logs", "log"]:
                if self.logs is None:
                    print("Invalid request. There was no query made previously.\n")
                else:
                    print("-"*50)
                    print("LOGS:")
                    for log in self.logs:
                        print(f"- {log}")
                    print("-"*50)
                continue
            
            self.response = self.agent.process_query(query)
            
            self.logs = self.response['log']
            
            print("\n" + "="*50)
            print(f"QUERY: {self.response['query']}")
            print(f"TOOL: {self.response['tool_used']}")
            print("-"*50)
            
            if self.response['tool_used'] == 'rag':
                print("RETRIEVED CHUNKS:")
                for i, chunk in enumerate(self.response['retrieved_chunks']):
                    print(f"\nChunk {i+1} (Source: {chunk['source']}, Score: {chunk['relevance_score']:.2f}):")
                    print(f"{chunk['content'][:200]}...")
                print("-"*50)
                print("RESULT:")
                print(self.response['result'].content)
                print("="*50 + "\n")
            else:
                print("RESULT:")
                print(self.response['result'])
                print("="*50 + "\n")

if __name__ == "__main__":
    obj = main()
    obj.cli_interface()