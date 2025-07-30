"""
Image Agent API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from ...agents.specialized.image_agent import ImageAgent
from ...agents.core.base_agent import AgentMetadata, AgentType
from ...agents.core.agent_factory import create_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])

class ImageGenerationRequest(BaseModel):
    content: Optional[str] = None
    blog_title: Optional[str] = None
    blog_id: Optional[str] = None
    outline: Optional[List[str]] = None
    style: str = "professional"
    count: int = 3

class ImageRegenerationRequest(BaseModel):
    style: str = "professional"
    blog_title: str
    content: str

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

@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    """
    Generate images based on blog content using the Image Agent.
    Can work with existing blog content or new content.
    """
    try:
        # If blog_id is provided, fetch the blog data
        if request.blog_id:
            logger.info(f"Fetching blog data for ID: {request.blog_id}")
            
            # Direct database query to avoid circular imports
            from ...config.database import db_config
            
            try:
                with db_config.get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT id, title, "contentMarkdown"
                        FROM "BlogPost" 
                        WHERE id = %s AND status != 'deleted'
                    """, (request.blog_id,))
                    row = cur.fetchone()
                    
                    if not row:
                        raise HTTPException(status_code=404, detail=f"Blog not found: {request.blog_id}")
                    
                    columns = [desc[0] for desc in cur.description]
                    row_dict = dict(zip(columns, row))
                    
                    content = row_dict["contentMarkdown"] or ""
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
        
        logger.info(f"Generating images for blog: {blog_title}")
        
        # Create image agent
        image_agent = create_agent(
            AgentType.IMAGE,
            metadata=AgentMetadata(
                agent_type=AgentType.IMAGE,
                name="ImageGenerationAgent",
                description="Generates images for blog content"
            )
        )
        
        # Prepare input context
        context = {
            "content": content,
            "blog_title": blog_title,
            "outline": request.outline or [],
            "style": request.style,
            "count": request.count
        }
        
        # Execute image agent
        result = image_agent.execute(context)
        
        if not result.success:
            raise HTTPException(
                status_code=500, 
                detail=f"Image generation failed: {result.error_message}"
            )
        
        # Convert result data to response format
        images = []
        for img_data in result.data.get("images", []):
            images.append(ImageData(**img_data))
        
        response = ImageGenerationResponse(
            success=True,
            images=images,
            prompts=result.data.get("prompts", []),
            style=result.data.get("style", request.style),
            count=len(images)
        )
        
        logger.info(f"Successfully generated {len(images)} images")
        return response
        
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@router.post("/regenerate/{image_id}", response_model=ImageData)
async def regenerate_image(image_id: str, request: ImageRegenerationRequest):
    """
    Regenerate a specific image with new parameters.
    """
    try:
        logger.info(f"Regenerating image {image_id}")
        
        # Create image agent
        image_agent = create_agent(
            AgentType.IMAGE,
            metadata=AgentMetadata(
                agent_type=AgentType.IMAGE,
                name="ImageRegenerationAgent",
                description="Regenerates specific images"
            )
        )
        
        # Prepare input context for regeneration
        context = {
            "content": request.content,
            "blog_title": request.blog_title,
            "outline": [],
            "style": request.style,
            "count": 1,
            "regenerate_id": image_id
        }
        
        # Execute image agent
        result = image_agent.execute(context)
        
        if not result.success:
            raise HTTPException(
                status_code=500, 
                detail=f"Image regeneration failed: {result.error_message}"
            )
        
        # Get the regenerated image
        images = result.data.get("images", [])
        if not images:
            raise HTTPException(status_code=500, detail="No image generated")
        
        regenerated_image = ImageData(**images[0])
        
        logger.info(f"Successfully regenerated image {image_id}")
        return regenerated_image
        
    except Exception as e:
        logger.error(f"Image regeneration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image regeneration failed: {str(e)}")

@router.get("/styles")
async def get_available_styles():
    """
    Get available image styles.
    """
    styles = [
        {"id": "professional", "name": "Profesional", "description": "Diseño limpio y corporativo"},
        {"id": "creative", "name": "Creativo", "description": "Estilo artístico y llamativo"},
        {"id": "minimalist", "name": "Minimalista", "description": "Diseño simple y elegante"},
        {"id": "modern", "name": "Moderno", "description": "Tendencias actuales de diseño"},
        {"id": "vintage", "name": "Vintage", "description": "Estilo retro y clásico"}
    ]
    return {"styles": styles}

@router.get("/sizes")
async def get_available_sizes():
    """
    Get available image sizes.
    """
    sizes = [
        {"id": "800x600", "name": "Estándar", "width": 800, "height": 600},
        {"id": "1200x630", "name": "Redes Sociales", "width": 1200, "height": 630},
        {"id": "1920x1080", "name": "HD", "width": 1920, "height": 1080},
        {"id": "400x400", "name": "Cuadrado", "width": 400, "height": 400}
    ]
    return {"sizes": sizes}

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify router is working."""
    return {"message": "Images router is working!"}

@router.get("/blogs")
async def get_available_blogs():
    """
    Get list of available blogs for image generation.
    """
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