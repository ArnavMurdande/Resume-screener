# AI-Powered Resume Screening with RAG 🚀

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React_19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Supabase](https://img.shields.io/badge/Database-Supabase_pgvector-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![Gemini](https://img.shields.io/badge/AI-Google_Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)

An intelligent Resume Screening tool that utilizes **Retrieval-Augmented Generation (RAG)** to provide deep insights into candidate suitability. It moves beyond keyword matching by understanding semantic context to calculate match scores and answer recruiter queries with verified evidence.

---

## 🏗️ Architecture Overview

The system implements a production-grade RAG pipeline to ensure accuracy and prevent LLM hallucinations.

Pipeline:
1. Extraction: Parse text from PDF/TXT and clean it.
2. Chunking: Split documents using RecursiveCharacterTextSplitter with overlap.
3. Embedding: Generate vector embeddings using Gemini text-embedding-004.
4. Vector Storage: Store embeddings in Supabase (PostgreSQL + pgvector with HNSW index).
5. Retrieval: Perform cosine similarity search to fetch top-k relevant chunks.
6. Generation: Gemini Flash generates answers grounded strictly in retrieved context.

### Architecture Diagram

![RAG Architecture Flow](https://github.com/ArnavMurdande/Resume-screener/blob/267dfa50aed8b6f3928e2091dd8f61ce90d72557/Diagram.png)


---

## ✨ Features

- Semantic Match Scoring with strengths, gaps, and executive summary
- RAG-powered contextual chat with session awareness
- Verified evidence with exact resume quotes
- Multi-key Gemini API rotation for rate-limit resilience
- Modern professional UI with glassmorphism design

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- Supabase project with pgvector enabled
- Google Gemini API key(s)

---

### Clone Repository

    git clone https://github.com/ArnavMurdande/Resume-screener.git
    cd Resume-screener

---

### Backend Setup

    cd backend
    python -m venv venv
    venv\Scripts\activate        (Windows)
    source venv/bin/activate     (Mac/Linux)
    pip install -r requirements.txt

Create a .env file inside backend/:

    GOOGLE_API_KEY_1=your_primary_gemini_key
    GOOGLE_API_KEY_2=your_secondary_gemini_key
    SUPABASE_URL=your_supabase_project_url
    SUPABASE_KEY=your_supabase_service_role_key

Run backend:

    uvicorn main:app --reload

---

### Frontend Setup

    cd ../Frontend
    npm install
    npm run dev

Frontend runs at:
http://localhost:5173

---

## 📋 API Documentation

POST /upload  
Uploads Resume or Job Description, chunks text, generates embeddings, and stores them in the vector DB.

Content-Type: multipart/form-data  
Fields:
- file
- doc_type (resume | jd)
- session_id

---

POST /analyze  

Request body:
    {
      "session_id": "string"
    }

Response:
    {
      "match_score": 78,
      "strengths": [],
      "gaps": [],
      "summary": "string",
      "highlights": []
    }

---

POST /chat  

Request body:
    {
      "question": "Does the candidate have React experience?",
      "history": [],
      "session_id": "string"
    }

---

## 🛠️ Tech Stack

Frontend:
- React 19
- TypeScript
- Tailwind CSS
- Lucide React

Backend:
- FastAPI
- LangChain
- pdfminer

Vector Database:
- Supabase (PostgreSQL + pgvector)

AI Models:
- Gemini Flash
- Gemini text-embedding-004

---

## 👨‍💻 Author

Arnav Murdande

GitHub: https://github.com/ArnavMurdande  
Portfolio: https://arnavmurdande.com  
LinkedIn: https://www.linkedin.com/in/arnav-murdande/
