from typing import Any
import os
from google import genai
from google.genai import types

class LLMService:
    def __init__(self):
        """Initialize the LLM service."""
        
        self._key = os.getenv("Gemini_API_Key", default = None)
        if self._key is None:
            raise Exception("Gemini API key not found. Please set the 'Gemini_API_Key' environment variable.")
        
        self.client = genai.Client(api_key = self._key)
    
    def generate_response(self, query: str, context_chunks: list[dict[str, Any]] = None):
        """Generate a response using the Ollama model via LangChain.
        
        Args:
            query: The user query.
            (optional) context_chunks: List of context chunks for the query. Default is None.
            
        Returns:
            Generated LLM response.
        """
        
        if context_chunks:
            context_text = "".join([chunk['content'] for chunk in context_chunks])
            response = self.client.models.generate_content(
                model = "gemini-2.0-flash-lite",
                config = types.GenerateContentConfig(
                    system_instruction = f'''You are an AI assistant with access to the following information. Use this information to answer the user's question. If the information doesn't contain the answer, say so. Do not make up information. If the question seems nonsensical, say so. CONTEXT INFORMATION:\n{context_text}'''
                    ),
                contents = query
                )
        else:
            response = self.client.models.generate_content(
                model = "gemini-2.0-flash-lite",
                config = types.GenerateContentConfig(
                    system_instruction = '''You are a helpful AI assistant. Answer the user's question based on your knowledge. If the question seems nonsensical, say so.'''
                    ),
                contents = query
                )
        return response