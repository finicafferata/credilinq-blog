#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Test Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Test server working"}

@app.get("/api/test")
def test():
    return {"message": "API test working"}

@app.get("/api/blogs")
def get_blogs():
    return [
        {"id": "test-1", "title": "Test Blog 1", "status": "draft", "created_at": "2025-07-30T15:30:00Z"},
        {"id": "test-2", "title": "Test Blog 2", "status": "published", "created_at": "2025-07-30T15:31:00Z"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 