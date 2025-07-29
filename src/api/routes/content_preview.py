"""
Content preview and editing interface for multi-format content generation.
Provides real-time preview and collaborative editing capabilities.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import json
import asyncio
from uuid import uuid4

from ...agents.specialized.content_repurposer import ContentPlatform, ContentTone
from ...core.auth import get_current_user, require_scope, APIKeyScope
from ...core.enhanced_exceptions import CustomHTTPException, ErrorCategory
from ...core.cache import cache

router = APIRouter()

class ContentPreviewRequest(BaseModel):
    """Request model for content preview."""
    content: str = Field(..., description="Content to preview")
    platform: ContentPlatform = Field(..., description="Platform for preview")
    preview_options: Dict[str, Any] = Field(default_factory=dict, description="Preview customization options")

class ContentEditRequest(BaseModel):
    """Request model for content editing."""
    content_id: str = Field(..., description="Content ID to edit")
    updated_content: str = Field(..., description="Updated content")
    platform: ContentPlatform = Field(..., description="Target platform")
    edit_notes: Optional[str] = Field(None, description="Notes about the edit")

class ContentValidationRequest(BaseModel):
    """Request model for content validation."""
    content: str = Field(..., description="Content to validate")
    platform: ContentPlatform = Field(..., description="Platform for validation")
    validation_rules: List[str] = Field(default_factory=list, description="Specific validation rules")

class PreviewResponse(BaseModel):
    """Response model for content preview."""
    preview_html: str
    platform_info: Dict[str, Any]
    validation_results: Dict[str, Any]
    suggestions: List[str] = Field(default_factory=list)

# WebSocket connection manager for real-time collaboration
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.content_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def send_to_session(self, message: str, session_id: str):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove disconnected connections
                    self.active_connections[session_id].remove(connection)

manager = ConnectionManager()

@router.post("/preview", response_model=PreviewResponse)
async def generate_content_preview(
    request: ContentPreviewRequest,
    current_user: Dict = Depends(get_current_user),
    _: bool = Depends(require_scope(APIKeyScope.BLOGS_READ))
) -> PreviewResponse:
    """
    Generate a visual preview of content formatted for a specific platform.
    
    This endpoint creates a realistic preview showing how the content would
    appear on the target platform, including formatting, hashtags, and styling.
    """
    
    try:
        # Generate platform-specific preview
        preview_html = await _generate_platform_preview(
            content=request.content,
            platform=request.platform,
            options=request.preview_options
        )
        
        # Get platform information
        platform_info = await _get_platform_info(request.platform)
        
        # Validate content for platform
        validation_results = await _validate_content(
            content=request.content,
            platform=request.platform
        )
        
        # Generate suggestions
        suggestions = await _generate_preview_suggestions(
            content=request.content,
            platform=request.platform,
            validation_results=validation_results
        )
        
        return PreviewResponse(
            preview_html=preview_html,
            platform_info=platform_info,
            validation_results=validation_results,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            error_code="PREVIEW_GENERATION_FAILED",
            message=f"Failed to generate content preview: {str(e)}",
            category=ErrorCategory.SYSTEM
        )

@router.post("/validate", response_model=Dict[str, Any])
async def validate_content(
    request: ContentValidationRequest,
    current_user: Dict = Depends(get_current_user),
    _: bool = Depends(require_scope(APIKeyScope.BLOGS_READ))
) -> Dict[str, Any]:
    """
    Validate content against platform-specific rules and best practices.
    
    This endpoint checks content for compliance with platform guidelines,
    optimal length, formatting, and engagement potential.
    """
    
    try:
        validation_results = await _comprehensive_content_validation(
            content=request.content,
            platform=request.platform,
            custom_rules=request.validation_rules
        )
        
        return {
            "success": True,
            "validation_results": validation_results,
            "overall_score": validation_results.get("overall_score", 0),
            "validated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            error_code="CONTENT_VALIDATION_FAILED",
            message=f"Content validation failed: {str(e)}",
            category=ErrorCategory.SYSTEM
        )

@router.post("/edit", response_model=Dict[str, Any])
async def edit_content(
    request: ContentEditRequest,
    current_user: Dict = Depends(get_current_user),
    _: bool = Depends(require_scope(APIKeyScope.BLOGS_WRITE))
) -> Dict[str, Any]:
    """
    Edit and update repurposed content with change tracking.
    
    This endpoint allows users to edit generated content while maintaining
    a history of changes and validating updates against platform requirements.
    """
    
    try:
        # Validate the updated content
        validation_results = await _validate_content(
            content=request.updated_content,
            platform=request.platform
        )
        
        # Save the edit
        edit_result = await _save_content_edit(
            content_id=request.content_id,
            updated_content=request.updated_content,
            platform=request.platform,
            user_id=current_user["user_id"],
            edit_notes=request.edit_notes
        )
        
        # Notify other collaborators if this is a shared session
        await _notify_collaborators(
            content_id=request.content_id,
            user_id=current_user["user_id"],
            change_type="edit",
            changes={"content": request.updated_content}
        )
        
        return {
            "success": True,
            "edit_id": edit_result["edit_id"],
            "validation_results": validation_results,
            "updated_at": datetime.now().isoformat(),
            "change_summary": edit_result["change_summary"]
        }
        
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            error_code="CONTENT_EDIT_FAILED",
            message=f"Failed to edit content: {str(e)}",
            category=ErrorCategory.SYSTEM
        )

@router.get("/editor/{content_id}")
async def get_content_editor(
    content_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the collaborative content editor interface.
    
    Returns an HTML interface for editing content with real-time collaboration,
    platform preview, and validation feedback.
    """
    
    editor_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CrediLinQ Content Editor</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .editor-container {{
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 400px;
                gap: 20px;
            }}
            .editor-panel {{
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .preview-panel {{
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                position: sticky;
                top: 20px;
                height: fit-content;
            }}
            .content-editor {{
                width: 100%;
                min-height: 300px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                padding: 16px;
                font-size: 16px;
                line-height: 1.5;
                resize: vertical;
                font-family: inherit;
            }}
            .content-editor:focus {{
                outline: none;
                border-color: #0066cc;
            }}
            .platform-selector {{
                display: flex;
                gap: 8px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .platform-btn {{
                padding: 8px 16px;
                border: 2px solid #e1e5e9;
                background: white;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .platform-btn.active {{
                background: #0066cc;
                color: white;
                border-color: #0066cc;
            }}
            .validation-status {{
                margin: 16px 0;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
            }}
            .validation-success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .validation-warning {{
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }}
            .validation-error {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f1aeb5;
            }}
            .preview-iframe {{
                width: 100%;
                height: 400px;
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                background: white;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin: 16px 0;
            }}
            .metric {{
                text-align: center;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 6px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #0066cc;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }}
            .suggestions {{
                margin-top: 20px;
            }}
            .suggestion {{
                padding: 8px 12px;
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                margin-bottom: 8px;
                border-radius: 0 4px 4px 0;
                font-size: 14px;
            }}
            .toolbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 1px solid #e1e5e9;
            }}
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .btn-primary {{
                background: #0066cc;
                color: white;
            }}
            .btn-primary:hover {{
                background: #0052a3;
            }}
            .btn-secondary {{
                background: #6c757d;
                color: white;
            }}
            .collaborators {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .collaborator-avatar {{
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: #0066cc;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="editor-container">
            <div class="editor-panel">
                <div class="toolbar">
                    <h2>Content Editor</h2>
                    <div class="collaborators">
                        <div class="collaborator-avatar">{current_user["email"][0].upper()}</div>
                        <button class="btn btn-primary" onclick="saveContent()">Save Changes</button>
                        <button class="btn btn-secondary" onclick="exportContent()">Export</button>
                    </div>
                </div>
                
                <div class="platform-selector">
                    <button class="platform-btn active" data-platform="linkedin_post">LinkedIn</button>
                    <button class="platform-btn" data-platform="twitter_thread">Twitter Thread</button>
                    <button class="platform-btn" data-platform="instagram_post">Instagram</button>
                    <button class="platform-btn" data-platform="facebook_post">Facebook</button>
                </div>
                
                <textarea class="content-editor" id="contentEditor" placeholder="Start editing your content here..."></textarea>
                
                <div class="validation-status" id="validationStatus">
                    <strong>✓ Content looks good!</strong> Ready to publish.
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="wordCount">0</div>
                        <div class="metric-label">Words</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="charCount">0</div>
                        <div class="metric-label">Characters</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="engagementScore">75</div>
                        <div class="metric-label">Engagement Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="readabilityScore">B+</div>
                        <div class="metric-label">Readability</div>
                    </div>
                </div>
            </div>
            
            <div class="preview-panel">
                <h3>Live Preview</h3>
                <iframe class="preview-iframe" id="previewFrame" src="about:blank"></iframe>
                
                <div class="suggestions" id="suggestions">
                    <h4>Suggestions</h4>
                    <div class="suggestion">
                        💡 Add a question at the end to boost engagement
                    </div>
                    <div class="suggestion">
                        🏷️ Consider adding 2-3 more relevant hashtags
                    </div>
                    <div class="suggestion">
                        📱 Content looks great for mobile viewing
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const contentEditor = document.getElementById('contentEditor');
            const platformBtns = document.querySelectorAll('.platform-btn');
            const wordCount = document.getElementById('wordCount');
            const charCount = document.getElementById('charCount');
            const validationStatus = document.getElementById('validationStatus');
            const previewFrame = document.getElementById('previewFrame');
            
            let currentPlatform = 'linkedin_post';
            let contentId = '{content_id}';
            let ws = null;
            
            // Initialize WebSocket connection
            function initWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${{protocol}}//${{window.location.host}}/api/content-preview/ws/${{contentId}}`);
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'content_update' && data.user_id !== '{current_user["user_id"]}') {{
                        // Handle collaborative updates
                        updateEditorFromCollaborator(data);
                    }}
                }};
            }}
            
            // Platform selection
            platformBtns.forEach(btn => {{
                btn.addEventListener('click', () => {{
                    platformBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentPlatform = btn.dataset.platform;
                    updatePreview();
                }});
            }});
            
            // Content editing
            contentEditor.addEventListener('input', () => {{
                updateMetrics();
                updatePreview();
                validateContent();
                
                // Send updates to collaborators
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{
                        type: 'content_update',
                        content: contentEditor.value,
                        platform: currentPlatform,
                        user_id: '{current_user["user_id"]}'
                    }}));
                }}
            }});
            
            function updateMetrics() {{
                const content = contentEditor.value;
                const words = content.trim().split(/\s+/).filter(w => w.length > 0).length;
                const chars = content.length;
                
                wordCount.textContent = words;
                charCount.textContent = chars;
                
                // Update engagement score based on content
                let score = 50;
                if (content.includes('?')) score += 15;
                if (words > 50) score += 10;
                if (content.includes('#')) score += 10;
                
                document.getElementById('engagementScore').textContent = Math.min(100, score);
            }}
            
            function updatePreview() {{
                // Generate platform-specific preview
                const content = contentEditor.value;
                let previewHTML = generatePlatformPreview(content, currentPlatform);
                
                previewFrame.srcdoc = previewHTML;
            }}
            
            function generatePlatformPreview(content, platform) {{
                const styles = {{
                    linkedin_post: 'background: #f3f2ef; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;',
                    twitter_thread: 'background: #15202b; color: white; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;',
                    instagram_post: 'background: #fafafa; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;',
                    facebook_post: 'background: #f0f2f5; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'
                }};
                
                return `
                    <html>
                        <head>
                            <style>
                                body {{ ${{styles[platform]}} }}
                                .post {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; }}
                                .author {{ display: flex; align-items: center; margin-bottom: 12px; }}
                                .author-avatar {{ width: 40px; height: 40px; border-radius: 50%; background: #0066cc; }}
                                .author-info {{ margin-left: 12px; }}
                                .author-name {{ font-weight: bold; }}
                                .post-content {{ line-height: 1.5; white-space: pre-wrap; }}
                                .hashtags {{ color: #0066cc; }}
                            </style>
                        </head>
                        <body>
                            <div class="post">
                                <div class="author">
                                    <div class="author-avatar"></div>
                                    <div class="author-info">
                                        <div class="author-name">{current_user.get("email", "User")}</div>
                                        <div>Just now</div>
                                    </div>
                                </div>
                                <div class="post-content">${{content.replace(/#(\w+)/g, '<span class="hashtags">#$1</span>')}}</div>
                            </div>
                        </body>
                    </html>
                `;
            }}
            
            function validateContent() {{
                const content = contentEditor.value;
                const length = content.length;
                
                // Platform-specific validation
                const limits = {{
                    linkedin_post: 3000,
                    twitter_thread: 280,
                    instagram_post: 2200,
                    facebook_post: 63206
                }};
                
                const limit = limits[currentPlatform];
                
                if (length === 0) {{
                    setValidationStatus('Start typing to see validation...', 'warning');
                }} else if (length > limit) {{
                    setValidationStatus(`Content is too long (${{length}}/${{limit}} characters)`, 'error');
                }} else if (length < limit * 0.1) {{
                    setValidationStatus('Content might be too short for good engagement', 'warning');
                }} else {{
                    setValidationStatus('Content looks good! Ready to publish.', 'success');
                }}
            }}
            
            function setValidationStatus(message, type) {{
                validationStatus.className = `validation-status validation-${{type}}`;
                validationStatus.innerHTML = `<strong>${{type === 'success' ? '✓' : type === 'error' ? '✗' : '⚠'}}</strong> ${{message}}`;
            }}
            
            function saveContent() {{
                fetch('/api/content-preview/edit', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer {current_user.get("token", "")}'
                    }},
                    body: JSON.stringify({{
                        content_id: contentId,
                        updated_content: contentEditor.value,
                        platform: currentPlatform,
                        edit_notes: 'Updated via editor interface'
                    }})
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        alert('Content saved successfully!');
                    }} else {{
                        alert('Failed to save content: ' + data.message);
                    }}
                }});
            }}
            
            function exportContent() {{
                const blob = new Blob([contentEditor.value], {{ type: 'text/plain' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `content-${{currentPlatform}}-${{new Date().toISOString().split('T')[0]}}.txt`;
                a.click();
                URL.revokeObjectURL(url);
            }}
            
            function updateEditorFromCollaborator(data) {{
                // Handle collaborative updates
                const currentPosition = contentEditor.selectionStart;
                contentEditor.value = data.content;
                contentEditor.setSelectionRange(currentPosition, currentPosition);
                updateMetrics();
                updatePreview();
            }}
            
            // Initialize
            initWebSocket();
            updateMetrics();
            validateContent();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=editor_html)

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time collaborative editing.
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast to all clients in the session
            await manager.send_to_session(data, session_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

# Helper functions for preview generation

async def _generate_platform_preview(
    content: str,
    platform: ContentPlatform,
    options: Dict[str, Any]
) -> str:
    """Generate HTML preview for specific platform."""
    
    platform_templates = {
        ContentPlatform.LINKEDIN_POST: _generate_linkedin_preview,
        ContentPlatform.TWITTER_THREAD: _generate_twitter_preview,
        ContentPlatform.INSTAGRAM_POST: _generate_instagram_preview,
        ContentPlatform.FACEBOOK_POST: _generate_facebook_preview
    }
    
    generator = platform_templates.get(platform, _generate_generic_preview)
    return await generator(content, options)

async def _generate_linkedin_preview(content: str, options: Dict[str, Any]) -> str:
    """Generate LinkedIn-style preview."""
    return f"""
    <div style="background: #f3f2ef; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="background: white; border-radius: 8px; padding: 16px; border: 1px solid #e6e6e6;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background: #0a66c2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    U
                </div>
                <div style="margin-left: 12px;">
                    <div style="font-weight: 600; font-size: 14px;">Professional User</div>
                    <div style="font-size: 12px; color: #666;">Just now • 🌐</div>
                </div>
            </div>
            <div style="line-height: 1.5; white-space: pre-wrap; font-size: 14px;">
                {content.replace('#', '<span style="color: #0a66c2;">#</span>')}
            </div>
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e6e6e6; display: flex; justify-content: space-between; color: #666; font-size: 12px;">
                <span>👍 💬 🔄 📤</span>
                <span>Like • Comment • Repost • Send</span>
            </div>
        </div>
    </div>
    """

async def _generate_twitter_preview(content: str, options: Dict[str, Any]) -> str:
    """Generate Twitter-style preview."""
    tweets = content.split('\n\n') if '\n\n' in content else [content]
    
    preview_html = '<div style="background: #15202b; padding: 20px; color: white; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
    
    for i, tweet in enumerate(tweets[:5]):  # Max 5 tweets preview
        preview_html += f"""
        <div style="background: #15202b; border: 1px solid #38444d; border-radius: 16px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; margin-bottom: 12px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #1d9bf0; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px;">
                    U
                </div>
                <div>
                    <div style="font-weight: bold;">User Name <span style="color: #8b98a5;">@username</span></div>
                    <div style="color: #8b98a5; font-size: 14px;">Just now</div>
                </div>
            </div>
            <div style="line-height: 1.5; font-size: 15px; white-space: pre-wrap;">
                {tweet.replace('#', '<span style="color: #1d9bf0;">#</span>')}
            </div>
            <div style="margin-top: 16px; display: flex; justify-content: space-between; color: #8b98a5; font-size: 13px;">
                <span>💬 12</span>
                <span>🔄 34</span>
                <span>❤️ 56</span>
                <span>📤</span>
            </div>
        </div>
        """
    
    preview_html += '</div>'
    return preview_html

async def _generate_instagram_preview(content: str, options: Dict[str, Any]) -> str:
    """Generate Instagram-style preview."""
    return f"""
    <div style="background: #fafafa; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="background: white; border: 1px solid #dbdbdb; border-radius: 8px; max-width: 400px;">
            <div style="padding: 16px; display: flex; align-items: center;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(45deg, #f09433 0%,#e6683c 25%,#dc2743 50%,#cc2366 75%,#bc1888 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 12px;">
                    U
                </div>
                <div style="font-weight: 600; font-size: 14px;">username</div>
            </div>
            <div style="background: #f0f0f0; height: 400px; display: flex; align-items: center; justify-content: center; color: #666;">
                📸 Image/Video Content
            </div>
            <div style="padding: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>❤️ 💬 📤</div>
                    <div>🔖</div>
                </div>
                <div style="font-weight: 600; font-size: 14px; margin-bottom: 8px;">username</div>
                <div style="line-height: 1.4; font-size: 14px; white-space: pre-wrap;">
                    {content.replace('#', '<span style="color: #00376b;">#</span>')}
                </div>
                <div style="color: #8e8e8e; font-size: 12px; margin-top: 8px;">Just now</div>
            </div>
        </div>
    </div>
    """

async def _generate_facebook_preview(content: str, options: Dict[str, Any]) -> str:
    """Generate Facebook-style preview."""
    return f"""
    <div style="background: #f0f2f5; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #1877f2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 12px;">
                    U
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 15px;">User Name</div>
                    <div style="color: #65676b; font-size: 13px;">Just now • 🌐</div>
                </div>
            </div>
            <div style="line-height: 1.5; font-size: 15px; color: #1c1e21; white-space: pre-wrap; margin-bottom: 12px;">
                {content.replace('#', '<span style="color: #1877f2;">#</span>')}
            </div>
            <div style="border-top: 1px solid #e4e6ea; padding-top: 8px; display: flex; justify-content: space-between; color: #65676b; font-size: 15px;">
                <span>👍 Like</span>
                <span>💬 Comment</span>
                <span>📤 Share</span>
            </div>
        </div>
    </div>
    """

async def _generate_generic_preview(content: str, options: Dict[str, Any]) -> str:
    """Generate generic preview for unsupported platforms."""
    return f"""
    <div style="background: #f5f5f5; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="background: white; border-radius: 8px; padding: 20px; border: 1px solid #e1e5e9;">
            <h3 style="margin-top: 0; color: #333;">Content Preview</h3>
            <div style="line-height: 1.6; color: #555; white-space: pre-wrap;">
                {content}
            </div>
        </div>
    </div>
    """

async def _get_platform_info(platform: ContentPlatform) -> Dict[str, Any]:
    """Get platform-specific information."""
    from ...agents.specialized.content_repurposer import PLATFORM_SPECS
    
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return {"error": "Platform not supported"}
    
    return {
        "name": platform.value.replace('_', ' ').title(),
        "max_length": spec.max_length,
        "optimal_length": spec.optimal_length,
        "character_limit": spec.character_limit,
        "preferred_tone": spec.preferred_tone.value,
        "supports_hashtags": spec.supports_hashtags,
        "supports_mentions": spec.supports_mentions,
        "engagement_hooks": spec.engagement_hooks
    }

async def _validate_content(content: str, platform: ContentPlatform) -> Dict[str, Any]:
    """Validate content for platform requirements."""
    from ...agents.specialized.content_repurposer import PLATFORM_SPECS
    
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return {"valid": False, "errors": ["Platform not supported"]}
    
    errors = []
    warnings = []
    
    # Length validation
    if len(content) > spec.max_length:
        errors.append(f"Content exceeds maximum length ({len(content)}/{spec.max_length} characters)")
    elif len(content) > spec.optimal_length:
        warnings.append(f"Content is longer than optimal ({len(content)}/{spec.optimal_length} characters)")
    
    # Content quality checks
    if len(content.strip()) == 0:
        errors.append("Content cannot be empty")
    
    if len(content.split()) < 10:
        warnings.append("Content might be too short for good engagement")
    
    # Platform-specific validations
    if platform == ContentPlatform.TWITTER_THREAD:
        if '\n\n' not in content and len(content) > 280:
            warnings.append("Consider breaking into multiple tweets")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "character_count": len(content),
        "word_count": len(content.split()),
        "optimal_range": f"{int(spec.optimal_length * 0.8)}-{spec.optimal_length}",
        "engagement_potential": "high" if len(errors) == 0 and len(warnings) == 0 else "medium" if len(errors) == 0 else "low"
    }

async def _comprehensive_content_validation(
    content: str,
    platform: ContentPlatform,
    custom_rules: List[str]
) -> Dict[str, Any]:
    """Perform comprehensive content validation."""
    
    basic_validation = await _validate_content(content, platform)
    
    # Additional validations
    score = 100
    
    # Engagement factors
    if '?' not in content:
        score -= 10
        basic_validation.setdefault('suggestions', []).append("Add a question to encourage engagement")
    
    if '#' not in content and platform != ContentPlatform.LINKEDIN_POST:
        score -= 15
        basic_validation.setdefault('suggestions', []).append("Consider adding relevant hashtags")
    
    # Readability
    sentences = content.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    if avg_sentence_length > 25:
        score -= 10
        basic_validation.setdefault('suggestions', []).append("Consider shorter sentences for better readability")
    
    basic_validation['overall_score'] = max(0, score)
    basic_validation['readability_grade'] = 'A' if score >= 90 else 'B' if score >= 75 else 'C' if score >= 60 else 'D'
    
    return basic_validation

async def _save_content_edit(
    content_id: str,
    updated_content: str,
    platform: ContentPlatform,
    user_id: str,
    edit_notes: Optional[str]
) -> Dict[str, Any]:
    """Save content edit with version history."""
    
    edit_id = str(uuid4())
    
    # In a real implementation, this would save to database
    edit_record = {
        "edit_id": edit_id,
        "content_id": content_id,
        "updated_content": updated_content,
        "platform": platform.value,
        "user_id": user_id,
        "edit_notes": edit_notes,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache the edit for immediate retrieval
    await cache.set("content_edits", edit_id, edit_record, ttl=3600)
    
    # Generate change summary
    change_summary = {
        "changes_made": "Content updated",
        "character_difference": len(updated_content),  # Would calculate actual diff
        "platform": platform.value
    }
    
    return {
        "edit_id": edit_id,
        "change_summary": change_summary,
        "saved_at": datetime.now().isoformat()
    }

async def _notify_collaborators(
    content_id: str,
    user_id: str,
    change_type: str,
    changes: Dict[str, Any]
):
    """Notify collaborators of content changes."""
    
    notification = {
        "type": "content_change",
        "content_id": content_id,
        "user_id": user_id,
        "change_type": change_type,
        "changes": changes,
        "timestamp": datetime.now().isoformat()
    }
    
    # Send to WebSocket connections
    await manager.send_to_session(json.dumps(notification), content_id)

async def _generate_preview_suggestions(
    content: str,
    platform: ContentPlatform,
    validation_results: Dict[str, Any]
) -> List[str]:
    """Generate suggestions for improving content."""
    
    suggestions = []
    
    # Based on validation results
    if validation_results.get('warnings'):
        suggestions.extend(validation_results['warnings'])
    
    # Platform-specific suggestions
    if platform == ContentPlatform.LINKEDIN_POST:
        if 'experience' not in content.lower() and 'insight' not in content.lower():
            suggestions.append("💡 Share a personal experience or insight to increase engagement")
    
    elif platform == ContentPlatform.TWITTER_THREAD:
        if content.count('\n\n') < 2:
            suggestions.append("🧵 Consider breaking this into multiple tweets for better thread engagement")
    
    elif platform == ContentPlatform.INSTAGRAM_POST:
        emoji_count = sum(1 for char in content if ord(char) > 127)
        if emoji_count < 3:
            suggestions.append("✨ Add more emojis to increase visual appeal on Instagram")
    
    # General suggestions
    if '?' not in content:
        suggestions.append("❓ Add a question at the end to encourage comments and engagement")
    
    if len(content.split()) < 50:
        suggestions.append("📝 Consider expanding the content with more details or examples")
    
    return suggestions[:5]  # Limit to 5 suggestions