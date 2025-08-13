"""
Debug Image Agent API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ImageGenerationRequest(BaseModel):
    content: Optional[str] = None
    blog_title: Optional[str] = None
    blog_id: Optional[str] = None
    outline: Optional[List[str]] = None
    style: str = "professional"
    count: int = 3

class ImageData(BaseModel):
    id: str
    prompt: str
    url: str
    alt_text: str
    style: str
    size: str

class ImageGenerationResponse(BaseModel):
    success: bool
    images: List[ImageData]
    prompts: List[str]
    style: str
    count: int

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify router is working."""
    return {"message": "Images debug router is working!"}

@router.get("/blogs")
async def get_available_blogs():
    """
    Get list of available blogs for image generation.
    """
    try:
        # Direct database query to get real blogs
        from ...config.database import db_config
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, status, "createdAt"
                FROM blog_posts 
                WHERE status != 'deleted' 
                ORDER BY "createdAt" DESC
            """)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            blogs = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert datetime to string
                row_dict['created_at'] = str(row_dict['createdAt'])
                del row_dict['createdAt']  # Remove the original key
                blogs.append({
                    "id": row_dict["id"],
                    "title": row_dict["title"],
                    "status": row_dict["status"],
                    "created_at": row_dict["created_at"]
                })
            
            return {"blogs": blogs}
            
    except Exception as e:
        # Fallback to test data if database query fails
        return {
            "blogs": [
                {
                    "id": "test-id-1",
                    "title": "Test Blog 1",
                    "status": "draft",
                    "created_at": "2025-07-29 20:00:00"
                },
                {
                    "id": "test-id-2", 
                    "title": "Test Blog 2",
                    "status": "published",
                    "created_at": "2025-07-29 19:00:00"
                }
            ]
        }

@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    """
    Generate images for content or existing blog.
    """
    try:
        logger.info(f"Generating images with request: {request}")
        
        # If blog_id is provided, fetch the blog data
        if request.blog_id:
            logger.info(f"Fetching blog data for ID: {request.blog_id}")
            
            # Direct database query to avoid circular imports
            from ...config.database import db_config
            
            try:
                with db_config.get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, title, content_markdown
                        FROM blog_posts 
                        WHERE id = %s AND status != 'deleted'
                    """, (request.blog_id,))
                    row = cur.fetchone()
                    
                    if not row:
                        raise HTTPException(status_code=404, detail=f"Blog not found: {request.blog_id}")
                    
                    columns = [desc[0] for desc in cur.description]
                    row_dict = dict(zip(columns, row))
                    
                    content = row_dict["content_markdown"] or ""
                    blog_title = row_dict["title"]
                    logger.info(f"Retrieved blog: {blog_title}")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error fetching blog {request.blog_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error fetching blog: {str(e)}")
        else:
            # Use provided content and title
            if not request.content or not request.blog_title:
                raise HTTPException(
                    status_code=400,
                    detail="Either blog_id or both content and blog_title must be provided"
                )
            content = request.content
            blog_title = request.blog_title
        
        # Generate mock images for now
        images = []
        prompts = []
        
        for i in range(request.count):
            image_id = f"img_{i+1}"
            prompt = f"Professional image for: {blog_title[:50]}..."
            
            # Create a very simple SVG data URL
            svg_content = f'<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg"><rect width="800" height="600" fill="#4F46E5"/><text x="400" y="300" font-family="Arial" font-size="24" fill="white" text-anchor="middle">Image {i+1}</text></svg>'
            
            # Encode SVG as data URL
            import base64
            svg_encoded = base64.b64encode(svg_content.encode()).decode()
            data_url = f"data:image/svg+xml;base64,{svg_encoded}"
            
            image_data = {
                "id": image_id,
                "prompt": prompt,
                "url": data_url,
                "alt_text": f"Generated image for: {prompt[:50]}...",
                "style": request.style,
                "size": "800x600"
            }
            images.append(ImageData(**image_data))
            prompts.append(prompt)
        
        logger.info(f"Generated {len(images)} images successfully")
        
        return ImageGenerationResponse(
            success=True,
            images=images,
            prompts=prompts,
            style=request.style,
            count=request.count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}") 