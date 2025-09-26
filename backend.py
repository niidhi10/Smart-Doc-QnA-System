# backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging

# import the ingestion function which returns a RetrievalQA chain
from ingest import ingest_document

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# allow Streamlit / local calls (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)

# Global variable that holds the latest in-memory RetrievalQA chain
qa_chain = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Save uploaded PDF, ingest it (build embeddings + FAISS in-memory),
    and set the global qa_chain to the RetrievalQA chain returned by ingest_document().
    """
    global qa_chain

    try:
        # save uploaded file
        file_path = os.path.join(DOCS_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"Saved uploaded file: {file_path}")

        # ingest_document(file_path) MUST return a RetrievalQA chain (as per your ingest.py)
        qa_chain = ingest_document(file_path)
        if qa_chain is None:
            raise RuntimeError("ingest_document returned None - check ingest.py")

        logging.info("Ingestion complete and qa_chain updated (in-memory).")
        return {"status": "ok", "message": f"{file.filename} uploaded and indexed successfully ✅"}

    except Exception as e:
        logging.exception("Upload / ingest failed")
        raise HTTPException(status_code=500, detail=f"Upload or ingestion failed: {str(e)}")


@app.post("/query")
async def query_doc(query: str = None):
    """
    Query endpoint. Accepts a query string as a query param or form param.
    Expects qa_chain to be set by /upload. Returns answer + sources.
    """
    global qa_chain

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No document indexed yet. Upload a PDF first.")

    try:
        # Run the RetrievalQA chain. The chain returns a dict-like result
        # Use the chain call interface (works for LangChain RetrievalQA)
        result = qa_chain({"query": query})

        # result may contain "result" or "answer" depending on chain: normalize
        answer = result.get("result") or result.get("answer") or str(result)

        # gather source documents metadata (if present)
        src_docs = result.get("source_documents", [])
        sources = []
        for d in src_docs:
            md = d.metadata if hasattr(d, "metadata") else {}
            sources.append({
                "doc": md.get("doc_name") or md.get("source") or "Unknown",
                "page": md.get("page_num") or md.get("page") or "N/A",
                "chunk_id": md.get("chunk_id", None)
            })

        return {"status": "ok", "answer": answer, "sources": sources}

    except Exception as e:
        logging.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")