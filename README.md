# RAG Resume Screener

A modern, AI-powered Resume Screening application built with React, FastAPI, Supabase (pgvector), and Gemini AI. This tool allows recruiters to upload a Candidate Resume and a Job Description (JD), then performs an intelligent gap analysis and answers questions about the candidate based on the documents.

## 🏗️ Architecture

```mermaid
graph LR
    User[Recruiter] --> Client[React Frontend (Vite)]
    Client -- Upload (PDF) --> API[FastAPI Backend]
    Client -- Chat/Analyze --> API
    API -- Extract Text --> PyPDF[pypdf]
    API -- Embeddings --> Gemini[Gemini Embeddings]
    API -- Store Vectors/Metadata --> DB[(Supabase pgvector)]
    API -- Query (RAG) --> DB
    API -- Generate Answer --> GeminiChat[Gemini Chat Model]
```

## 🚀 Key Features

- **User Isolation:** Uses Session IDs to ensure data privacy between different users/sessions.
- **Smart Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` for optimal context retrieval.
- **Intelligent Analysis:** Generates a structured Match Report with Score, Strengths, and Gaps.
- **Interactive Chat:** Ask questions about the specific candidate's fit for the specific JD.
- **Modern UI:** Glassmorphism design with Dark Mode by default.

## 🛠️ Setup & Installation

### Prerequisites

- Node.js (v18+)
- Python (3.9+)
- Supabase Project (with `pgvector` enabled)
- Google Gemini API Key

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
GOOGLE_API_KEY=your_gemini_api_key
```

Run the server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

## 📚 API Endpoints

| Method   | Endpoint   | Description                                                        |
| :------- | :--------- | :----------------------------------------------------------------- |
| **POST** | `/upload`  | Upload PDF (Resume/JD). Requires `file`, `doc_type`, `session_id`. |
| **POST** | `/chat`    | Chat with the context. Body: `question`, `history`, `session_id`.  |
| **POST** | `/analyze` | Generate Match Report. Body: `session_id`.                         |

## 🛡️ Database Schema (Supabase)

Make sure you run the SQL RPC setup provided to enable vector functionality and session filtering.
