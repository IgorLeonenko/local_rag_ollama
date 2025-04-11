import re
import smtplib
import uuid
from email.mime.text import MIMEText

import streamlit as st
from phi.agent import Agent
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.model.message import Message
from phi.model.ollama import Ollama
from phi.vectordb.qdrant import Qdrant
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient, models

# Streamlit app title
st.title("RAG Agent with Qdrant")

# Initialize Qdrant and embedder
qdrant_client = QdrantClient(host="localhost", port=6333)
embedder = OllamaEmbedder()

# Define the collection name
COLLECTION_NAME = ""
EMBEDDING_DIMENSION = 4096


# Ensure Qdrant collection exists
def ensure_collection_exists():
    collection = qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
    if not collection:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "size": EMBEDDING_DIMENSION,
                "distance": models.Distance.COSINE,
            },
        )
        st.success(f"Collection '{COLLECTION_NAME}' created successfully!")
    else:
        st.success(f"Collection '{COLLECTION_NAME}' loaded successfully.")


ensure_collection_exists()


# Check if a file has already been uploaded
def is_file_uploaded(file_name):
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=[0] * EMBEDDING_DIMENSION,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_name", match=models.MatchValue(value=file_name)
                )
            ]
        ),
        limit=1,
    )
    return len(results) > 0


# loading in batches
def batch_points(points, batch_size):
    for i in range(0, len(points), batch_size):
        yield points[i : i + batch_size]


# preprocess text
def preprocess_text(text):
    # Remove extra whitespace and blank lines
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# Function to process and upload PDF files to Qdrant
def upload_pdf_to_qdrant(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    pdf_text = preprocess_text(pdf_text)

    # Split the text into chunks
    chunks = [pdf_text[i : i + 500] for i in range(0, len(pdf_text), 500)]

    # Generate embeddings and prepare points
    points = []
    for idx, chunk in enumerate(chunks):
        embedding = embedder.get_embedding(chunk)
        uu_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{uploaded_file.name}_chunk_{idx}")
        points.append(
            models.PointStruct(
                id=f"{uu_id}",
                vector=embedding,
                payload={
                    "content": chunk,
                    "document_name": uploaded_file.name,
                    "chunk_index": idx,
                },
            )
        )

    # Upload points in batches to Qdrant
    for batch in batch_points(points, 200):
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
    st.success(f"Uploaded {len(points)} chunks from '{uploaded_file.name}' to Qdrant!")


# PDF upload functionality
st.header("Upload PDFs to Enrich Knowledge Base")
uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if is_file_uploaded(uploaded_file.name):
            st.warning(f"File '{uploaded_file.name}' is already uploaded. Skipping...")
        else:
            upload_pdf_to_qdrant(uploaded_file)

if st.checkbox("List all uploaded files"):
    results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)[0]
    uploaded_files = {point.payload["document_name"] for point in results}
    st.write("### Uploaded Files")
    for file_name in uploaded_files:
        st.write(f"- {file_name}")

# Initialize Llama Agent with Qdrant Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls=[],
    vector_db=Qdrant(
        collection=COLLECTION_NAME, url="http://localhost:6333/", embedder=embedder
    ),
)

llama_agent = Agent(
    name="Llama Agent", model=Ollama(id="llama3.2"), knowledge=knowledge_base
)

# Query functionality
st.header("Ask Questions About Uploaded Documents")
query = st.text_input("Ask a question:", "")

if "response_content" not in st.session_state:
    st.session_state.response_content = ""

if st.button("Submit Query"):
    if query.strip():
        # Query the agent
        with st.spinner("Thinking..."):
            response = llama_agent.model.response([Message(role="user", content=query)])
            response_content = response.content
            if response:
                st.session_state.response_content = response.content
            else:
                st.session_state.response_content = ""
    else:
        st.write("Ask something!")

if st.session_state.response_content:
    st.write("### Agent Response:")
    st.write(st.session_state.response_content)

    if st.checkbox("Send response via email"):
        email_recipient = st.text_input("Recipient Email Address")
        email_subject = st.text_input("Email Subject", "Response from RAG Agent")
        if st.button("Send Response via Email"):
            with st.spinner("Sending..."):
                try:
                    # SMTP configuration
                    smtp_server = "smtp.gmail.com"
                    smtp_port = 587
                    smtp_user = ""  # Replace with your email
                    smtp_password = (
                        ""  # Replace with your app password
                    )

                    # Create email message
                    msg = MIMEText(st.session_state.response_content)
                    msg["Subject"] = email_subject
                    msg["From"] = smtp_user
                    msg["To"] = email_recipient

                    # Send email
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                        server.sendmail(smtp_user, email_recipient, msg.as_string())

                    st.success(f"Email sent successfully to {email_recipient}!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
else:
    st.write("No response generated.")
