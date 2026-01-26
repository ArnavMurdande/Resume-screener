
# ... (Imports remain similar but ensure we have everything)
import os
import io
import json
import traceback
import re
import random
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
from pdfminer.high_level import extract_text
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import ast

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- API KEY MANAGEMENT ---
env_keys = {}
for key, val in os.environ.items():
    if key.startswith("GOOGLE_API_KEY"):
        clean_val = val.strip().strip('"').strip("'")
        env_keys[key] = clean_val

sorted_key_names = sorted(env_keys.keys())
API_KEYS = [env_keys[k] for k in sorted_key_names]

if not API_KEYS:
    raise ValueError("No GOOGLE_API_KEY variables found in environment!")

def get_embedding_with_retry(text: str):
    """Generates embedding with retry logic across all keys."""
    keys_to_try = API_KEYS.copy()
    random.shuffle(keys_to_try)
    
    for i, key in enumerate(keys_to_try):
        try:
            model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
            return model.embed_query(text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise e
    raise HTTPException(status_code=429, detail="All API keys exhausted for embeddings.")

def get_batch_embeddings_with_retry(texts: List[str]) -> List[List[float]]:
    keys_to_try = API_KEYS.copy()
    random.shuffle(keys_to_try)
    
    for i, key in enumerate(keys_to_try):
        try:
            model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
            return model.embed_documents(texts)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise e
    raise HTTPException(status_code=429, detail="All API keys exhausted for batch embeddings.")

def clean_and_parse_json(content: Any) -> Dict:
    """
    Robust JSON parser that handles:
    1. Standard JSON strings.
    2. Markdown code blocks.
    3. Stringified Python dictionaries (Gemini edge case).
    """
    try:
        # 1. If already a dict, return it
        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                 return clean_and_parse_json(content["text"])
            return content

        # 2. If list, join (Gemini chunks)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str): text_parts.append(item)
                elif isinstance(item, dict) and "text" in item: text_parts.append(item["text"])
            content = "".join(text_parts) if text_parts else str(content)

        # 3. Ensure it's a string
        content = str(content).strip()

        # 4. CRITICAL FIX: Handle Stringified Python Dicts
        if content.startswith("{") and "'type':" in content:
            try:
                parsed_obj = ast.literal_eval(content)
                if isinstance(parsed_obj, dict) and "text" in parsed_obj:
                    return clean_and_parse_json(parsed_obj["text"])
            except Exception:
                pass

        # 5. Clean Markdown
        content = re.sub(r"```(json)?", "", content, flags=re.IGNORECASE)
        content = content.replace("```", "").strip()
        
        # 6. Parse
        return json.loads(content)

    except Exception as e:
        print(f"[JSON Parse Error] Content: {str(content)[:100]}... Error: {e}")
        return {}


# --- DATA MODELS ---

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    session_id: str

class AnalyzeRequest(BaseModel):
    session_id: str

# --- EXTRACTION LOGIC ---

async def extract_structured_data(session_id: str, doc_type: str, filename: str):
    """
    Performs internal RAG to extract structured data from the uploaded document.
    Updates the chunks' metadata with this extraction.
    """
    try:
        query = "technical skills, programming languages, years of experience, education degree, job requirements"
        if doc_type == "jd":
            query = "required skills, experience years, education requirements, job responsibilities"
            
        # 1. Embed query
        query_embedding = get_embedding_with_retry(query)
        
        # 2. Retrieve relevant chunks (Internal RAG)
        response = supabase.rpc(
            "match_resume_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.30,
                "match_count": 8,
                "filter": {"session_id": session_id, "type": doc_type, "filename": filename}
            }
        ).execute()
        
        matched_chunks = response.data
        if not matched_chunks:
            print(f"[Extraction] No chunks found for {filename}")
            return {}

        context_text = "\n\n".join([c.get("content", "") for c in matched_chunks])
        
        # 3. LLM Extraction
        prompt = f"""You are a Data Extraction Assistant. Extract the following JSON from the text.
        
        TEXT:
        {context_text}
        
        REQUIRED JSON STRUCTURE:
        {{
            "skills": ["<skill1>", "<skill2>", ...], (Normalize to lowercase strings)
            "experience_years": <number>, (Total years mentioned, use 0 if none)
            "education_level": "<Highest Degree Mentioned>" (e.g. "Bachelors", "Masters", "PhD", or "None")
        }}
        
        Output ONLY the valid JSON."""
        
        # Retry logic for generation
        keys_to_try = API_KEYS.copy()
        
        extracted_data = {"skills": [], "experience_years": 0, "education_level": "None"}
        
        for key in keys_to_try:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=key, model_kwargs={"response_mime_type": "application/json"})
                result = llm.invoke(prompt)
                
                # Use Helper
                extracted_data = clean_and_parse_json(result.content)
                if extracted_data:
                    break
            except Exception as e:
                print(f"[Extraction] Error with key {key[:5]}...: {e}")
                continue
                
        # 4. Update Metadata of ALL chunks for this file
        
        new_metadata = {
            "filename": filename,
            "type": doc_type,
            "session_id": session_id,
            "extracted": extracted_data
        }
        
        supabase.table("resume_chunks").update({"metadata": new_metadata}) \
            .eq("metadata->>session_id", session_id) \
            .eq("metadata->>filename", filename) \
            .execute()
            
        print(f"[Extraction] Successfully updated metadata for {filename}")
        return extracted_data

    except Exception as e:
        print(f"[Extraction] Failed: {e}")
        traceback.print_exc()
        return {}

# --- UTILS ---

def get_pdf_text(pdf_file: bytes) -> str:
    try:
        return extract_text(io.BytesIO(pdf_file))
    except Exception:
        return ""

def clean_text(text: str) -> str:
    # Remove surrogate characters
    return text.encode('utf-8', 'ignore').decode('utf-8')

def normalize_text(text: str) -> str:
    """Cleans text for better embedding quality."""
    text = clean_text(text)
    # Standardize bullet points
    text = re.sub(r"[•●▪➢➣]", "- ", text)
    # Collapse multiple newlines/spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_latest_document_text(doc_type: str, session_id: str) -> str:
    try:
        latest_record = supabase.table("resume_chunks").select("metadata").eq("metadata->>type", doc_type).eq("metadata->>session_id", session_id).order("id", desc=True).limit(1).execute()
        if not latest_record.data: return ""
        filename = latest_record.data[0]['metadata'].get('filename')
        if not filename: return ""
        chunks = supabase.table("resume_chunks").select("content").eq("metadata->>filename", filename).eq("metadata->>session_id", session_id).order("id").execute()
        return "".join([c['content'] for c in chunks.data]) if chunks.data else ""
    except:
        return ""

def get_document_metadata(doc_type: str, session_id: str) -> Dict:
    """Fetches the extracted metadata for the latest document."""
    try:
        latest = supabase.table("resume_chunks").select("metadata").eq("metadata->>type", doc_type).eq("metadata->>session_id", session_id).limit(1).execute()
        if latest.data:
            return latest.data[0]['metadata'].get('extracted', {})
        return {}
    except:
        return {}

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    session_id: str = Form(...)
):
    try:
        content = await file.read()
        
        # 1. Detect file type (PDF vs TXT)
        if file.filename.lower().endswith(".txt"):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")
        else:
            text = get_pdf_text(content)
            
        if not text.strip():
             raise HTTPException(status_code=400, detail="Could not extract text from file.")

        # 1.5. Normalize (CRITICAL FIX)
        text = normalize_text(text)

        chunks = split_text(text)

        # 2. CRITICAL FIX: Explicitly wipe OLD data for this session/type
        # We use .eq() chains because .match() can be unreliable for deletion in some client versions.
        print(f"[Upload] Wiping old {doc_type} chunks for session {session_id}...")
        try:
            supabase.table("resume_chunks").delete() \
                .eq("metadata->>session_id", session_id) \
                .eq("metadata->>type", doc_type) \
                .execute()
        except Exception as e:
            print(f"[Warning] Delete step failed (might be empty): {e}")

        # 3. Embed & Insert New Chunks
        embeddings = get_batch_embeddings_with_retry(chunks)
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "content": chunk,
                "metadata": {
                    "filename": file.filename,
                    "type": doc_type,
                    "session_id": session_id,
                },
                "embedding": embeddings[i]
            })

        if data_to_insert:
            supabase.table("resume_chunks").insert(data_to_insert).execute()
        
        # 4. Trigger Extraction (Only for Resume usually, but safe to run for both)
        await extract_structured_data(session_id, doc_type, file.filename)
        
        return {"message": "Processed successfully", "filename": file.filename}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_resume(request: ChatRequest):
    try:
        q, h, sid = request.question, request.history, request.session_id
        
        # 1. STATE VALIDATION (Ensure documents exist)
        def check_exists(dtype):
            r = supabase.table("resume_chunks").select("id").eq("metadata->>session_id", sid).eq("metadata->>type", dtype).limit(1).execute()
            return len(r.data) > 0

        has_resume = check_exists("resume")
        has_jd = check_exists("jd")
        
        if not has_resume or not has_jd:
            missing = []
            if not has_resume: missing.append("Resume")
            if not has_jd: missing.append("Job Description")
            return {"answer": f"⚠️ I cannot answer yet because the following documents are missing: {', '.join(missing)}. Please upload them to start."}

        # 2. DUAL VECTOR SEARCH (Resume & JD)
        q_embed = get_embedding_with_retry(q)
        
        resume_res = supabase.rpc("match_resume_chunks", {
            "query_embedding": q_embed, 
            "match_threshold": 0.30, 
            "match_count": 8, 
            "filter": {"session_id": sid, "type": "resume"}
        }).execute()
        
        jd_res = supabase.rpc("match_resume_chunks", {
            "query_embedding": q_embed, 
            "match_threshold": 0.30, 
            "match_count": 5, 
            "filter": {"session_id": sid, "type": "jd"} 
        }).execute()
        
        # 3. CONTEXT CONSTRUCTION WITH FALLBACK (Fixes "Missing Context" Bug)
        resume_chunks = [c['content'] for c in resume_res.data] if resume_res.data else []
        jd_chunks = [c['content'] for c in jd_res.data] if jd_res.data else []

        # Fallback: If vector search results are empty but docs exist, fetch FULL text
        # This prevents RAG from failing on general questions like "Skills?" or "Summary?"
        if not resume_chunks:
            print("[Chat] Vector search miss for Resume. Fetching full text fallback.")
            full_resume = get_latest_document_text("resume", sid)
            if full_resume: resume_chunks = [full_resume[:20000]] # Limit fallback size
            
        if not jd_chunks:
            print("[Chat] Vector search miss for JD. Fetching full text fallback.")
            full_jd = get_latest_document_text("jd", sid)
            if full_jd: jd_chunks = [full_jd[:10000]] # Limit fallback size

        context = f"""
        [RESUME CONTENT]
        {" ... ".join(resume_chunks)}
        
        [JOB DESCRIPTION CONTENT]
        {" ... ".join(jd_chunks)}
        """
        
        if len(context) > 25000: context = context[:25000]

        # 4. SYSTEM PROMPT WIRING (Fixes "Context Not Passed" Bug)
        # We explicitly include {{context}} in the system message.
        system_template = """You are a Helpful AI Recruiter Assistant.
        use the following pieces of context to answer the user's question.
        
        CONTEXT:
        {context}
        
        GUIDELINES:
        - Answer based STRICTLY on the provided context.
        - If the answer isn't in the context, say "I don't see that mentioned in the documents."
        - Do not make up facts.
        - Be concise, professional, and direct.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder("history"),
            ("human", "{q}")
        ])
        
        lc_hist = [HumanMessage(content=m.get("content","")) if m.get("role")=="user" else AIMessage(content=m.get("content","")) for m in h]
        
        # Retry loop for LLM
        for key in API_KEYS:
            try:
                chat = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=key)
                chain = prompt | chat | StrOutputParser()
                # Pass context explicitly
                ans = chain.invoke({"context": context, "history": lc_hist, "q": q})
                
                if not ans or not ans.strip():
                    ans = "I analyzed the documents but couldn't generate a specific answer. Could you rephrase your question?"
                
                return {"answer": ans}
            except Exception as e:
                if "429" in str(e): continue
                raise e
        raise HTTPException(status_code=429, detail="Gemini Quota Exceeded.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_match(request: AnalyzeRequest):
    try:
        sid = request.session_id
        
        # 1. Get JD Text & Embedding
        jd_text = get_latest_document_text("jd", sid)
        if not jd_text: raise HTTPException(404, detail="Missing JD")
        
        jd_query = jd_text[:2000]
        jd_embed = get_embedding_with_retry(jd_query)
        
        # 2. Vector Search (Resume Chunks)
        vec_res = supabase.rpc("match_resume_chunks", {
            "query_embedding": jd_embed, "match_threshold": 0.30, "match_count": 15, "filter": {"session_id": sid, "type": "resume"}
        }).execute()
        
        matched_chunks = vec_res.data or []
        if not matched_chunks: raise HTTPException(404, detail="Missing Resume")
        
        # 3. Hybrid Metrics Calculation (Defensive)
        try:
            # A. Vector Score (Avg of top 5 sims)
            top_5_sims = [c['similarity'] for c in matched_chunks[:5]]
            vector_score = (sum(top_5_sims) / len(top_5_sims)) * 100 if top_5_sims else 0
            vector_score = min(100, max(0, vector_score)) # Clamp
            
            # B. Skill Overlap
            resume_meta = get_document_metadata("resume", sid) or {}
            jd_meta = get_document_metadata("jd", sid) or {}
            
            # Safely get lists (handle None explicitly)
            r_skills_list = resume_meta.get("skills") if resume_meta else []
            if not isinstance(r_skills_list, list): r_skills_list = []
            
            j_skills_list = jd_meta.get("skills") if jd_meta else []
            if not isinstance(j_skills_list, list): j_skills_list = []

            resume_skills = set(str(s).lower() for s in r_skills_list)
            jd_skills = set(str(s).lower() for s in j_skills_list)
            
            overlap_score = 0
            if jd_skills:
                intersection = resume_skills.intersection(jd_skills)
                overlap_score = (len(intersection) / len(jd_skills)) * 100
            
            print(f"[Hybrid] Vector: {vector_score:.2f}, Overlap: {overlap_score:.2f} (Resume: {len(resume_skills)} skills, JD: {len(jd_skills)} skills)")
        except Exception as metric_error:
            print(f"[Warning] Metric calculation failed: {metric_error}")
            vector_score = 0
            overlap_score = 0

        # 4. LLM Analysis
        resume_context = "\n\n".join([f"[CHUNK {i}] {c['content']}" for i,c in enumerate(matched_chunks, 1)])
        
        prompt = f"""You are a Strategic Recruiter. Analyze the match between the Resume and JD.
        
        ## CALCULATED METRICS
        - Vector Similarity Score: {vector_score:.1f}/100
        - Skill Overlap Score: {overlap_score:.1f}/100
        
        ## RESUME CONTEXT
        {resume_context}
        
        ## JD CONTEXT
        {jd_text[:5000]}
        
        ## TASK
        Provide a JSON report.
        1. 'match_score': Your honest 0-100 assessment based on the text.
        2. 'strengths' & 'gaps': Key bullet points.
        3. 'highlights': Extract 3-5 EXACT QUOTES from the resume that best demonstrate the candidate's fit for this specific JD.
        
        STRUCTURE:
        {{
            "match_score": <number>,
            "strengths": ["<strength_1>", "<strength_2>"],
            "gaps": ["<gap_1>", "<gap_2>"],
            "highlights": [
                {{ "quote": "<exact_text_from_resume>", "relevance": "<why_it_matters>" }},
                {{ "quote": "<exact_text_from_resume>", "relevance": "<why_it_matters>" }}
            ],
            "summary": "<executive_summary>"
        }}
        """
        
        llm_response_json = {}
        for key in API_KEYS:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=key, model_kwargs={"response_mime_type": "application/json"})
                res = llm.invoke(prompt)
                
                # USE HELPER HERE
                llm_response_json = clean_and_parse_json(res.content)
                if llm_response_json:
                    break
            except Exception as e:
                print(f"Error LLM: {e}")
                continue
        
        if not llm_response_json:
             # Fallback if LLM fails completely
             llm_response_json = {
                 "match_score": 0, 
                 "strengths": ["Analysis Unavailable due to high load"],
                 "gaps": ["Please try again"], 
                 "summary": "Could not generate qualitative analysis."
             }
             
        # 5. Compute Hybrid Score
        try:
            llm_score = llm_response_json.get("match_score", 0)
            if not isinstance(llm_score, (int, float)): 
                llm_score = 0
            
            # Weights: Vector 40%, Skill 40%, LLM 20%
            final_score = (vector_score * 0.4) + (overlap_score * 0.4) + (llm_score * 0.2)
            final_score = int(min(100, max(0, round(final_score))))
            
            llm_response_json["match_score"] = final_score
        except Exception as calc_error:
             print(f"[Warning] Final score calc failed: {calc_error}")
             # Ensure we return a valid integer even if calc fails
             llm_response_json["match_score"] = int(llm_score) if isinstance(llm_score, (int, float)) else 0
        
        # Store Report
        try:
            # Explicitly cast match_score to int to prevent type errors
            score_int = int(llm_response_json.get("match_score", 0))
            
            data_payload = {
                "session_id": sid, 
                "match_score": score_int, 
                "analysis_json": llm_response_json
            }
            
            print(f"[History] Attempting to save report for session {sid}...")
            result = supabase.table("analysis_reports").insert(data_payload).execute()
            print(f"[History] Saved successfully. ID: {result.data[0].get('id') if result.data else 'Unknown'}")
            
        except Exception as db_err:
            print(f"CRITICAL DB ERROR: Failed to save history report! Reason: {db_err}")
            # Do not raise exception here, return the response so the UI still works
        
        return llm_response_json

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Fetches analysis history for a session."""
    try:
        response = supabase.table("analysis_reports").select("*").eq("session_id", session_id).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []
