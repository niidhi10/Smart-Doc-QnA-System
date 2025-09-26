Smart Document Q&A System

A RAG (Retrieval-Augmented Generation) system to ask questions from uploaded PDFs. It extracts text, chunks it, generates embeddings with HuggingFace, stores vectors in FAISS, and retrieves relevant chunks to answer queries using a local LLM (Flan-T5).

Features:

Upload PDFs and index instantly

Ask questions in natural language

Answers sourced from the document with page references


Tech Stack:

Frontend: Streamlit

Backend: FastAPI + Uvicorn

NLP / AI: LangChain, FAISS, HuggingFace Transformers (flan-t5-base, all-MiniLM-L6-v2)

PDF Handling: PyPDF, PyPDFLoader
