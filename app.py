# Streamlit Frontend
import streamlit as st
import requests
import os

BACKEND_URL = "http://127.0.0.1:8000"  # FastAPI runs here

st.set_page_config(page_title="RAG Q&A App", page_icon="📘")
st.title("📘 Smart Document Q&A")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save temporarily and send to backend
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    with open(uploaded_file.name, "rb") as f:
        response = requests.post(f"{BACKEND_URL}/upload", files={"file": f})

    os.remove(uploaded_file.name)
    if response.status_code == 200:
        st.success(f"Uploaded {uploaded_file.name} ✅")
    else:
        st.error("Upload failed ❌")

st.write("Now you can ask questions 👇")

# Step 2: Ask Query
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    with st.spinner("Thinking..."):
        response = requests.post(f"{BACKEND_URL}/query", params={"query": user_input})
    
    if response.status_code == 200:
        data = response.json()
        st.markdown(f"**Answer:** {data['answer']}")

        # Show retrieved sources
        with st.expander("Sources"):
            for src in data["sources"]:
                st.write(f"- {src['doc']} (p.{src['page']})")
    else:
        st.error("Something went wrong while querying ❌")