# RAG Agent System

This project implements a Retrieval-Augmented Generation (RAG) agent that can answer user queries by retrieving relevant information from a set of documents and generating responses using a Large Language Model (LLM). It supports both command-line and Streamlit web interfaces.

## Features

- Loads and chunks documents from various formats: `.txt`, `.csv`, `.json`, `.pdf`
- Stores document embeddings in a vector store (ChromaDB)
- Retrieves relevant document chunks for a given query
- Uses Gemini LLM for response generation, with or without context
- CLI and Streamlit interfaces for interaction

## Project Structure

```
.
├── data/                   # Source documents for retrieval
├── src/
│   ├── agent.py            # RAG agent logic
│   ├── app.py              # Streamlit web interface
│   ├── document_loader.py  # Document loading and chunking
│   ├── embeddings.py       # Embedding and vector store logic
│   ├── llm.py              # LLM service integration
│   ├── main.py             # CLI interface and entry point
│   └── retrieval.py        # Retriever logic
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── chatbot_responses.pdf   # Sample questions and chatbot responses
```

## Example Query

- "What is your flagship product?"
- "List some classic books and their authors."
- "Tell me about the NVIDIA RTX 5090."

## Logs

You can view logs of the last query by typing `logs` in the CLI or using the logs feature in the Streamlit sidebar.

## Citations

This project was developed using the following resources:

- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/introduction)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs/)
- [Streamlit API Reference](https://docs.streamlit.io/develop/api-reference)
- [ChatGPT](https://chat.openai.com/)
- [Claude](https://claude.ai/)