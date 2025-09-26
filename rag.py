from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA

INDEX_DIR = "faiss_index"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_local_llm():
    model_id = "google/flan-t5-base"   # small & efficient
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)

def faiss_retriever(embeddings, k=4, metadata_filter=None):
    from langchain_community.vectorstores import FAISS
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # Basic filtering: run similarity first, then post-filter by metadata dict keys
    if metadata_filter:
        base_get = retriever.get_relevant_documents
        def filtered_get(query):
            docs = base_get(query)
            keep = []
            for d in docs:
                ok = all(d.metadata.get(k) == v for k, v in metadata_filter.items())
                if ok: keep.append(d)
            return keep or docs  # fall back if filter removes everything
        retriever.get_relevant_documents = filtered_get
    return retriever

TOP_K = 4  # try 3, 4, 5
#retriever = faiss_retriever(embeddings, k=TOP_K)  # or hybrid_retriever(...)

def main():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = get_local_llm()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        docs = retriever.get_relevant_documents(query)
        context = " ".join([d.page_content for d in docs])

        prompt = f"Answer the question based on context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
        response = llm(prompt)
        print("\nResponse:", response)


SYSTEM_INSTRUCTIONS = """You are a careful assistant.
Use ONLY the provided CONTEXT to answer.
If the answer is not in the context, say "I don't know".
Cite sources like [doc_name p.page_num] inline where relevant.
Be concise (3-6 sentences)."""

def build_prompt(context, question):
    return f"""{SYSTEM_INSTRUCTIONS}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def get_rag_chain(top_k=4):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = get_local_llm() 

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", 
        return_source_documents=True
    )
    return qa_chain