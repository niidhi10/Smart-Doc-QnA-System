import os, argparse
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from rag import get_local_llm  # ✅ import your LLM loader

INDEX_DIR = "faiss_index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_document(file_path, chunk_size=600, chunk_overlap=80):
    print(f"Loading Document: {file_path}...    ")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Add useful metadata
    doc_name = os.path.basename(file_path)
    try:
        title = PdfReader(file_path).metadata.title or doc_name
    except Exception:
        title = doc_name
    for i, d in enumerate(pages):
        d.metadata["doc_name"] = doc_name
        d.metadata["title"] = title
        d.metadata["page_num"] = d.metadata.get("page", i + 1)

    print(f"Splitting (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    for j, c in enumerate(chunks):
        c.metadata["chunk_id"] = j

    print("Generating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # Always create fresh vectorstore in-memory
    vs = FAISS.from_documents(chunks, embeddings)

    # Build retriever + QA chain
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = get_local_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain   # ✅ now backend gets the latest chain

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="docs/sample.pdf")
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=80)
    args = ap.parse_args()
    if os.path.exists(args.file):
        ingest_document(args.file, args.chunk_size, args.chunk_overlap)
    else:
        print(f"File not found: {args.file}")