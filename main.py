from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import app as blog_agent_app
from agent import BlogWriterState
from setup_retriever import process_and_embed_document
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import uuid
import json
import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

api = FastAPI(
    title="Credilinq.ai Content Platform API",
    description="A scalable, persistent blog and knowledge management platform powered by multi-agent AI and Supabase.",
    version="3.0.0"
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://credilinq-blog-git-main-fini-cafferatas-projects.vercel.app",
        "https://credilinq-blog.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BlogCreateRequest(BaseModel):
    title: str
    company_context: str
    content_type: str = "blog"  # "blog" or "linkedin"

class BlogEditRequest(BaseModel):
    content_markdown: str

class BlogReviseRequest(BaseModel):
    instruction: str
    text_to_revise: str

class BlogSummary(BaseModel):
    id: str
    title: str
    status: str
    created_at: str

class BlogDetail(BlogSummary):
    content_markdown: str
    initial_prompt: dict

@api.post("/blogs", response_model=BlogSummary)
def create_blog(request: BlogCreateRequest):
    """
    Generate a new blog post using the multi-agent workflow and store it in Supabase.
    """
    agent_input = {
        "blog_title": request.title,
        "company_context": request.company_context,
        "content_type": request.content_type
    }
    result = blog_agent_app.invoke(agent_input)
    final_post = result.get("final_post", "")
    initial_prompt = json.dumps(agent_input)
    blog_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    data = {
        "id": blog_id,
        "title": request.title,
        "content_markdown": final_post,
        "initial_prompt": initial_prompt,
        "status": "draft",
        "created_at": created_at
    }
    resp = supabase.table("blog_posts").insert(data).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp}")
    return BlogSummary(id=blog_id, title=request.title, status="draft", created_at=created_at)

@api.get("/blogs", response_model=List[BlogSummary])
def list_blogs():
    """
    List all blog posts stored in Supabase.
    """
    resp = supabase.table("blog_posts").select("id, title, status, created_at").order("created_at", desc=True).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp}")
    # Try to get data from resp.data or resp['data'] or resp directly
    data = getattr(resp, 'data', None) or resp.get('data', None) or resp
    return [BlogSummary(**row) for row in data]

@api.get("/blogs/{post_id}", response_model=BlogDetail)
def get_blog(post_id: str):
    """
    Retrieve a single blog post by ID.
    """
    resp = supabase.table("blog_posts").select("*").eq("id", post_id).single().execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise HTTPException(status_code=404, detail="Blog post not found")
    row = getattr(resp, 'data', None) or resp.get('data', None) or resp
    return BlogDetail(
        id=row["id"],
        title=row["title"],
        status=row["status"],
        created_at=row["created_at"],
        content_markdown=row["content_markdown"],
        initial_prompt=json.loads(row["initial_prompt"])
    )

@api.put("/blogs/{post_id}", response_model=BlogDetail)
def edit_blog(post_id: str, request: BlogEditRequest):
    """
    Manually edit the content of a blog post.
    """
    resp = supabase.table("blog_posts").update({"content_markdown": request.content_markdown, "status": "edited"}).eq("id", post_id).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp}")
    return get_blog(post_id)

@api.post("/blogs/{post_id}/revise")
def revise_blog(post_id: str, request: BlogReviseRequest):
    """
    Use the LLM to revise a section of a blog post based on user instruction.
    """
    prompt = f"Revise the following text according to this instruction: '{request.instruction}'.\n\nText:\n{request.text_to_revise}"
    llm = blog_agent_app.graph.nodes["writer"].__globals__["llm"]
    response = llm.invoke(prompt)
    return {"revised_text": response.content}

@api.post("/documents/upload")
def upload_document(file: UploadFile = File(...), document_title: Optional[str] = None):
    """
    Upload a document to Supabase Storage and process it for RAG.
    """
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[-1]
    storage_path = f"{file_id}{file_ext}"
    file_content = file.file.read()
    supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(storage_path, file_content)
    doc_data = {
        "id": file_id,
        "title": document_title or file.filename,
        "storage_path": storage_path,
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    }
    resp = supabase.table("documents").insert(doc_data).execute()
    if not resp or getattr(resp, "status_code", 200) >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp}")
    process_and_embed_document(storage_path, file_id)
    return {"document_id": file_id, "storage_path": storage_path}

@api.get("/")
def root():
    return {"message": "Credilinq.ai Content Platform API is running", "status": "healthy"}

# For Vercel deployment
handler = api 