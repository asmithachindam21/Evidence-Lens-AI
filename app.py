import os
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

MODEL_NAME = "openai/gpt-oss-120b:fastest"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Evidence Lens AI",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🔎 Evidence Lens AI")

st.subheader(
    "AI-Powered Evidence Analysis and Retrieval"
)

st.write(
    "Upload a PDF evidence document and ask questions. "
    "The AI answers using the retrieved evidence."
)


# =========================================================
# API KEY CHECK
# =========================================================

if not HF_TOKEN:

    st.error(
        "Hugging Face API token not found."
    )

    st.info(
        "Create a .env file and add: HF_TOKEN=your_token"
    )

    st.stop()


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# CHROMADB
# =========================================================

@st.cache_resource
def load_collection():

    chroma_client = chromadb.PersistentClient(
        path="./evidence_lens_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="evidence_documents"
    )

    return collection


collection = load_collection()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    complete_text = ""

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            complete_text += (
                f"\n[Page {page_number + 1}]\n"
            )

            complete_text += text

    return complete_text


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text, chunk_size):

    words = text.split()

    chunks = []

    for start in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# =========================================================
# STORE EVIDENCE
# =========================================================

def store_evidence(
    chunks,
    filename
):

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []

    metadata = []

    for i in range(len(chunks)):

        ids.append(
            f"{filename}_{i}"
        )

        metadata.append(
            {
                "source": filename,
                "chunk_number": i
            }
        )

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata
    )


# =========================================================
# RETRIEVE EVIDENCE
# =========================================================

def retrieve_evidence(
    question,
    top_k
):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return documents, metadatas


# =========================================================
# HUGGING FACE API
# =========================================================

def ask_huggingface(
    question,
    evidence
):

    evidence_text = "\n\n".join(
        evidence
    )

    prompt = f"""
You are Evidence Lens AI.

Your job is to analyze evidence provided by the user.

IMPORTANT RULES:

1. Use ONLY the evidence provided below.
2. Do not invent facts.
3. Do not assume information that is not present.
4. If the evidence does not answer the question,
   clearly say that the evidence is insufficient.
5. Explain your answer clearly.
6. Identify the evidence supporting your answer.
7. Do not present guesses as facts.

EVIDENCE:
{evidence_text}

QUESTION:
{question}

Return the response in this format:

ANSWER:
Give a clear answer.

EVIDENCE FOUND:
Explain the important evidence supporting the answer.

CONCLUSION:
Give a short conclusion.

CONFIDENCE:
High, Medium, or Low.
"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=data,
        timeout=120
    )

    if response.status_code != 200:

        raise Exception(
            f"Hugging Face API Error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Evidence Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=1000,
    value=500,
    step=100
)

top_k = st.sidebar.slider(
    "Evidence Pieces to Retrieve",
    min_value=1,
    max_value=5,
    value=3
)


# =========================================================
# UPLOAD PDF
# =========================================================

st.header("📄 Upload Evidence")

uploaded_file = st.file_uploader(
    "Choose an evidence PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    st.info(
        f"Selected file: {uploaded_file.name}"
    )

    if st.button(
        "🔍 Process Evidence"
    ):

        with st.spinner(
            "Reading evidence document..."
        ):

            text = extract_pdf_text(
                uploaded_file
            )

        if not text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

        else:

            chunks = create_chunks(
                text,
                chunk_size
            )

            with st.spinner(
                "Creating evidence embeddings..."
            ):

                store_evidence(
                    chunks,
                    uploaded_file.name
                )

            st.session_state[
                "evidence_processed"
            ] = True

            st.session_state[
                "filename"
            ] = uploaded_file.name

            st.success(
                "Evidence processed successfully!"
            )

            st.metric(
                "Evidence Chunks",
                len(chunks)
            )


# =========================================================
# EVIDENCE DASHBOARD
# =========================================================

if st.session_state.get(
    "evidence_processed",
    False
):

    st.header("📊 Evidence Lens Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Document",
        st.session_state.get(
            "filename",
            "Unknown"
        )
    )

    col2.metric(
        "Stored Evidence",
        collection.count()
    )

    col3.metric(
        "AI Model",
        "Qwen 2.5"
    )


# =========================================================
# ASK QUESTION
# =========================================================

st.header("💬 Ask Evidence Lens AI")

question = st.text_input(
    "Ask a question about the evidence:"
)


# =========================================================
# ANALYZE QUESTION
# =========================================================

if question:

    if not st.session_state.get(
        "evidence_processed",
        False
    ):

        st.warning(
            "Please upload and process an evidence PDF first."
        )

    else:

        with st.spinner(
            "Searching relevant evidence..."
        ):

            documents, metadatas = retrieve_evidence(
                question,
                top_k
            )

        if not documents:

            st.warning(
                "No relevant evidence was found."
            )

        else:

            try:

                with st.spinner(
                    "Evidence Lens AI is analyzing..."
                ):

                    answer = ask_huggingface(
                        question,
                        documents
                    )

                st.subheader(
                    "🤖 Evidence Lens Analysis"
                )

                st.write(
                    answer
                )

                # =================================================
                # SUPPORTING EVIDENCE
                # =================================================

                st.subheader(
                    "📚 Retrieved Evidence"
                )

                for i, document in enumerate(
                    documents
                ):

                    source = metadatas[i].get(
                        "source",
                        "Unknown"
                    )

                    chunk_number = metadatas[i].get(
                        "chunk_number",
                        "Unknown"
                    )

                    with st.expander(
                        f"Evidence {i + 1} | "
                        f"{source} | "
                        f"Chunk {chunk_number}"
                    ):

                        st.write(
                            document
                        )

            except Exception as error:

                st.error(
                    "Could not generate the AI analysis."
                )

                st.code(
                    str(error)
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Evidence Lens AI | "
    "PDF + Embeddings + ChromaDB + Hugging Face"
)