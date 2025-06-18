# PDF RAG Assistant

An AI-powered PDF Question Answering application using **Retrieval-Augmented Generation (RAG)** with Google's **Gemini** model. Upload PDF files and get accurate, context-aware answers from them in a beautiful chat interface.

## 🚀 Live App

👉 [Click here to try the app](https://buildfastwithai.streamlit.app/)

## 📸 Preview

![Screenshot 2025-05-19 212647](https://github.com/user-attachments/assets/3d01d2bc-e0f7-4521-bc8e-bd63147b124d)


---

## 🧠 Features

- 🔍 **Ask Questions From PDFs** – Upload multiple PDF documents and ask questions about their content.
- 🧩 **RAG Pipeline** – Combines embedding-based retrieval with large language model generation.
- 🤖 **Gemini (Google Generative AI)** – Uses Gemini 1.5 Flash via the Google API.
- 🧠 **LangChain Integration** – For managing chains, prompts, embeddings, and memory.
- 💾 **FAISS Vector Store** – Efficient similarity search on chunked document data.
- 🧾 **Conversation History + CSV Export** – View and export your chat history with timestamps.
- 🌐 **Streamlit Interface** – Clean, minimal UI with chatbot-like interaction.

---

## 🧰 Tech Stack

| Technology        | Purpose                                      |
|-------------------|----------------------------------------------|
| `Python`          | Core programming language                    |
| `Streamlit`       | Web interface / deployment                   |
| `PyPDF2`          | PDF reading and text extraction              |
| `LangChain`       | RAG logic (prompting, chaining, embedding)   |
| `Google Generative AI` | Embedding & generative LLM (Gemini)   |
| `FAISS`           | Vector similarity search                     |
| `Pandas`          | CSV export of conversation history           |
| `Base64`          | Encoding downloadable data                   |
| `datetime`        | Timestamping interactions                    |

---

## 🛠️ Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/divishjindal19/Build_Fast_With_AI.git
cd Build_Fast_With_AI

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

Build_Fast_With_AI/
├── app.py               # Main Streamlit app
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation
├── LICENSE              # License (MIT)
└── .gitignore           # Files ignored by Git
