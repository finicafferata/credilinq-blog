"""
Content-First Deliverable Workflow
Transforms the task-centric approach into a deliverable-centric one
where agents produce structured content objects instead of task results.
"""

import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
# Import LangGraph components with version compatibility
from src.agents.core.langgraph_compat import StateGraph, END
from langchain_core.messages import SystemMessage
from src.core.llm_client import create_llm
from dotenv import load_dotenv
from datetime import datetime

# Import our database service and models
from ..core.database_service import DatabaseService
from ..core.content_deliverable_service import content_service

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Content-First State Schema ---
class ContentDeliverableState(TypedDict):
    # Campaign context
    campaign_id: str
    briefing: Annotated[Dict[str, Any], "Campaign briefing information"]
    content_strategy: Annotated[Dict[str, Any], "Content strategy for the campaign"]
    
    # Content narrative coordination
    narrative_theme: str
    story_arc: List[str]
    target_deliverables: List[Dict[str, Any]]  # What content pieces we need to create
    
    # Generated deliverables (the actual content)
    deliverables: Annotated[List[Dict[str, Any]], "Generated content deliverables"]
    
    # Content relationships and flow
    content_relationships: Annotated[Dict[str, List[str]], "How content pieces reference each other"]
    
    # Workflow metadata
    current_deliverable_index: int
    total_deliverables: int
    narrative_coordinator_notes: str

# --- LLM Setup ---
llm = create_llm(model="gemini-1.5-pro", temperature=0.7, api_key=GEMINI_API_KEY)

# --- Content-First Node Functions ---

def narrative_coordinator_node(state: ContentDeliverableState):
    """
    Creates the overall narrative strategy and plans deliverables as a cohesive story
    """
    logger.info("🎭 --- NARRATIVE COORDINATOR: Planning Content Story ---")
    
    briefing = state['briefing']
    
    prompt = f"""
    You are the Narrative Coordinator, responsible for creating a cohesive content strategy that tells one unified story.
    
    CAMPAIGN BRIEFING:
    - Objective: {briefing.get('marketing_objective', 'Not specified')}
    - Target Audience: {briefing.get('target_audience', 'Not specified')}
    - Channels: {briefing.get('channels', [])}
    - Tone: {briefing.get('desired_tone', 'Professional')}
    - Company Context: {briefing.get('company_context', 'Not specified')}
    
    Create a narrative strategy that includes:
    1. A central narrative theme that connects all content
    2. A story arc that flows across content pieces
    3. Specific deliverables with clear relationships
    
    Return your response as JSON:
    {{
        "narrative_theme": "The central story theme",
        "story_arc": ["Beginning hook", "Development", "Climax", "Resolution"],
        "target_deliverables": [
            {{
                "title": "Content piece title",
                "content_type": "blog_post|social_media_post|email_campaign",
                "platform": "blog|linkedin|twitter|email",
                "narrative_order": 1,
                "key_messages": ["message 1", "message 2"],
                "content_brief": "What this piece should accomplish",
                "references_to": []  // Will be populated as content is created
            }}
        ]
    }}
    
    Focus on creating 3-5 high-quality, interconnected pieces rather than many disconnected tasks.
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    try:
        narrative_strategy = json.loads(response.content)
        logger.info(f"📖 Created narrative strategy: {narrative_strategy['narrative_theme']}")
        logger.info(f"📋 Planned {len(narrative_strategy['target_deliverables'])} deliverables")
        
        return {
            **state,
            "narrative_theme": narrative_strategy['narrative_theme'],
            "story_arc": narrative_strategy['story_arc'],
            "target_deliverables": narrative_strategy['target_deliverables'],
            "current_deliverable_index": 0,
            "total_deliverables": len(narrative_strategy['target_deliverables']),
            "deliverables": [],
            "content_relationships": {},
            "narrative_coordinator_notes": f"Narrative theme: {narrative_strategy['narrative_theme']}"
        }
        
    except json.JSONDecodeError:
        logger.error("❌ Failed to parse narrative strategy JSON")
        # Fallback strategy
        return {
            **state,
            "narrative_theme": "Expert insights and industry leadership",
            "story_arc": ["Industry challenge", "Solution approach", "Success stories", "Future outlook"],
            "target_deliverables": [
                {
                    "title": f"Content piece for {briefing.get('marketing_objective', 'business goals')}",
                    "content_type": "blog_post",
                    "platform": "blog",
                    "narrative_order": 1,
                    "key_messages": ["Industry expertise", "Practical solutions"],
                    "content_brief": "Create valuable content addressing the marketing objective",
                    "references_to": []
                }
            ],
            "current_deliverable_index": 0,
            "total_deliverables": 1,
            "deliverables": [],
            "content_relationships": {},
            "narrative_coordinator_notes": "Using fallback narrative strategy"
        }

def content_researcher_node(state: ContentDeliverableState):
    """
    Researches content for the current deliverable with narrative context
    """
    logger.info("🔍 --- CONTENT RESEARCHER: Gathering Information ---")
    
    current_deliverable = state['target_deliverables'][state['current_deliverable_index']]
    narrative_theme = state['narrative_theme']
    
    # Use the existing research system but focus on the deliverable
    db_service = DatabaseService()
    
    try:
        # Search for relevant information based on the content brief
        research_query = f"{current_deliverable['content_brief']} {narrative_theme}"
        research_results = db_service.search_documents(research_query, limit=3)
        
        research_content = ""
        if research_results:
            research_content = "\n\n".join([doc.get('content', '') for doc in research_results])
        
        logger.info(f"📚 Gathered research for: {current_deliverable['title']}")
        
        return {
            **state,
            "current_research": research_content or "General knowledge base research"
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Research failed, using general approach: {e}")
        return {
            **state,
            "current_research": f"Research for {current_deliverable['title']} based on industry best practices"
        }

def content_creator_node(state: ContentDeliverableState):
    """
    Creates actual content deliverables (not task results)
    """
    logger.info("✍️ --- CONTENT CREATOR: Generating Deliverable ---")
    
    current_deliverable = state['target_deliverables'][state['current_deliverable_index']]
    narrative_theme = state['narrative_theme']
    briefing = state['briefing']
    research = state.get('current_research', '')
    
    # Create context-aware prompt based on content type
    content_type = current_deliverable['content_type']
    platform = current_deliverable['platform']
    
    if content_type == 'blog_post':
        content_prompt = create_blog_post_prompt(current_deliverable, narrative_theme, briefing, research, state)
    elif content_type == 'social_media_post':
        content_prompt = create_social_media_prompt(current_deliverable, narrative_theme, briefing, research, state)
    elif content_type == 'email_campaign':
        content_prompt = create_email_prompt(current_deliverable, narrative_theme, briefing, research, state)
    else:
        content_prompt = create_generic_content_prompt(current_deliverable, narrative_theme, briefing, research, state)
    
    response = llm.invoke([SystemMessage(content=content_prompt)])
    
    # Create the deliverable object
    deliverable = {
        "id": f"deliverable_{state['current_deliverable_index']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": current_deliverable['title'],
        "content": response.content,
        "summary": extract_summary(response.content),
        "content_type": content_type,
        "platform": platform,
        "narrative_order": current_deliverable['narrative_order'],
        "key_messages": current_deliverable['key_messages'],
        "created_by": "ContentCreatorAgent",
        "word_count": len(response.content.split()),
        "reading_time": max(1, len(response.content.split()) // 250),  # Approximate reading time
        "metadata": {
            "narrative_theme": narrative_theme,
            "content_brief": current_deliverable['content_brief'],
            "creation_timestamp": datetime.now().isoformat()
        }
    }
    
    # Add to deliverables list
    new_deliverables = state['deliverables'] + [deliverable]
    
    logger.info(f"📄 Created deliverable: {deliverable['title']} ({deliverable['word_count']} words)")
    
    return {
        **state,
        "deliverables": new_deliverables,
        "current_deliverable_index": state['current_deliverable_index'] + 1
    }

def narrative_connector_node(state: ContentDeliverableState):
    """
    Creates connections and references between content pieces for narrative flow
    """
    logger.info("🔗 --- NARRATIVE CONNECTOR: Linking Content Pieces ---")
    
    deliverables = state['deliverables']
    
    if len(deliverables) < 2:
        logger.info("   Not enough deliverables to create connections")
        return state
    
    # Create relationships between content pieces
    relationships = {}
    
    for i, deliverable in enumerate(deliverables):
        deliverable_id = deliverable['id']
        relationships[deliverable_id] = []
        
        # Connect to previous piece (narrative flow)
        if i > 0:
            prev_deliverable = deliverables[i-1]
            relationships[deliverable_id].append(f"Continues narrative from '{prev_deliverable['title']}'")
        
        # Connect to next piece (if exists)
        if i < len(deliverables) - 1:
            next_deliverable = deliverables[i+1]
            relationships[deliverable_id].append(f"Leads to '{next_deliverable['title']}'")
        
        # Thematic connections (same key messages)
        for j, other_deliverable in enumerate(deliverables):
            if i != j:
                shared_messages = set(deliverable['key_messages']) & set(other_deliverable['key_messages'])
                if shared_messages:
                    relationships[deliverable_id].append(f"Reinforces themes with '{other_deliverable['title']}'")
                    break  # Only add one thematic connection to avoid clutter
    
    logger.info(f"🔗 Created {sum(len(refs) for refs in relationships.values())} content relationships")
    
    return {
        **state,
        "content_relationships": relationships
    }

def content_storage_node(state: ContentDeliverableState):
    """
    Stores deliverables in the database and creates narrative structure
    """
    logger.info("💾 --- CONTENT STORAGE: Saving Deliverables ---")
    
    db_service = DatabaseService()
    campaign_id = state['campaign_id']
    narrative_theme = state['narrative_theme']
    story_arc = state['story_arc']
    deliverables = state['deliverables']
    relationships = state['content_relationships']
    
    try:
        # Create content narrative record
        narrative_record = {
            "campaign_id": campaign_id,
            "title": f"Content Narrative for Campaign",
            "description": f"Cohesive content strategy with theme: {narrative_theme}",
            "narrative_theme": narrative_theme,
            "key_story_arc": story_arc,
            "content_flow": {
                "deliverables_order": [d['id'] for d in deliverables],
                "relationships": relationships,
                "story_progression": story_arc
            },
            "total_pieces": len(deliverables),
            "completed_pieces": len(deliverables)
        }
        
        # Store deliverables in database
        stored_deliverable_ids = []
        for deliverable in deliverables:
            db_deliverable = {
                "title": deliverable['title'],
                "content": deliverable['content'],
                "summary": deliverable['summary'],
                "content_type": deliverable['content_type'],
                "campaign_id": campaign_id,
                "narrative_order": deliverable['narrative_order'],
                "key_messages": deliverable['key_messages'],
                "platform": deliverable['platform'],
                "word_count": deliverable['word_count'],
                "reading_time": deliverable['reading_time'],
                "created_by": deliverable['created_by'],
                "metadata": deliverable['metadata']
            }
            
            # Store in database using our fixed service
            try:
                created_deliverable = content_service.create_content(
                    title=deliverable['title'],
                    content=deliverable['content'],
                    campaign_id=campaign_id,
                    content_type=deliverable['content_type'],
                    summary=deliverable['summary'],
                    narrative_order=deliverable['narrative_order'],
                    key_messages=deliverable['key_messages'],
                    target_audience=deliverable.get('target_audience'),
                    tone=deliverable.get('tone'),
                    platform=deliverable['platform'],
                    metadata=deliverable['metadata']
                )
                deliverable_id = created_deliverable.id if created_deliverable else None
            except Exception as e:
                logger.error(f"❌ Failed to create deliverable '{deliverable['title']}': {e}")
                deliverable_id = None
            
            if deliverable_id:
                stored_deliverable_ids.append(deliverable_id)
                logger.info(f"💾 Stored deliverable: {deliverable['title']}")
            else:
                logger.warning(f"⚠️ Skipped storing deliverable: {deliverable['title']}")
        
        # Store narrative record
        db_service.create_content_narrative(narrative_record)
        
        logger.info(f"✅ Successfully stored {len(deliverables)} deliverables with narrative structure")
        
        return {
            **state,
            "stored_deliverable_ids": stored_deliverable_ids,
            "storage_complete": True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to store deliverables: {e}")
        return {
            **state,
            "storage_error": str(e),
            "storage_complete": False
        }

# --- Helper Functions ---

def create_blog_post_prompt(deliverable, narrative_theme, briefing, research, state):
    """Create optimized prompt for blog post generation"""
    return f"""
    Create a comprehensive blog post that serves as a key piece in our content narrative.
    
    CONTENT DELIVERABLE BRIEF:
    Title: {deliverable['title']}
    Purpose: {deliverable['content_brief']}
    Key Messages: {', '.join(deliverable['key_messages'])}
    Narrative Order: {deliverable['narrative_order']} of {state['total_deliverables']}
    
    NARRATIVE CONTEXT:
    Theme: {narrative_theme}
    Story Arc Position: {state['story_arc'][min(deliverable['narrative_order']-1, len(state['story_arc'])-1)]}
    
    CAMPAIGN CONTEXT:
    Company: {briefing.get('company_context', 'Not specified')}
    Audience: {briefing.get('target_audience', 'Business professionals')}
    Tone: {briefing.get('desired_tone', 'Professional')}
    
    RESEARCH FOUNDATION:
    {research}
    
    REQUIREMENTS:
    - Write a complete, ready-to-publish blog post
    - Use markdown formatting with proper headers
    - 1500-2500 words for comprehensive coverage
    - Include engaging introduction and strong conclusion
    - Naturally incorporate all key messages
    - Reference the narrative theme throughout
    - Create content that flows with the overall story arc
    - Include actionable insights and practical advice
    
    Write the complete blog post now:
    """

def create_social_media_prompt(deliverable, narrative_theme, briefing, research, state):
    """Create optimized prompt for social media content"""
    return f"""
    Create engaging social media content that continues our narrative story.
    
    CONTENT DELIVERABLE BRIEF:
    Title: {deliverable['title']}
    Platform: {deliverable['platform']}
    Purpose: {deliverable['content_brief']}
    Key Messages: {', '.join(deliverable['key_messages'])}
    Narrative Position: {deliverable['narrative_order']} of {state['total_deliverables']}
    
    NARRATIVE CONTEXT:
    Theme: {narrative_theme}
    Story Connection: {state['story_arc'][min(deliverable['narrative_order']-1, len(state['story_arc'])-1)]}
    
    CAMPAIGN CONTEXT:
    Company: {briefing.get('company_context', 'Not specified')}
    Audience: {briefing.get('target_audience', 'Business professionals')}
    Tone: {briefing.get('desired_tone', 'Professional')}
    
    RESEARCH FOUNDATION:
    {research}
    
    PLATFORM REQUIREMENTS:
    - LinkedIn: Professional tone, 800-1200 words, industry insights
    - Twitter: Concise, engaging, 2-3 tweets in a thread
    - Create content that references or builds on other pieces in the narrative
    
    Write the complete social media content now:
    """

def create_email_prompt(deliverable, narrative_theme, briefing, research, state):
    """Create optimized prompt for email content"""
    return f"""
    Create compelling email content that advances our narrative story.
    
    CONTENT DELIVERABLE BRIEF:
    Title: {deliverable['title']}
    Purpose: {deliverable['content_brief']}
    Key Messages: {', '.join(deliverable['key_messages'])}
    Narrative Position: {deliverable['narrative_order']} of {state['total_deliverables']}
    
    NARRATIVE CONTEXT:
    Theme: {narrative_theme}
    Story Element: {state['story_arc'][min(deliverable['narrative_order']-1, len(state['story_arc'])-1)]}
    
    EMAIL REQUIREMENTS:
    - Compelling subject line
    - Engaging opening that hooks the reader
    - Clear value proposition
    - Actionable content
    - Strong call-to-action
    - Professional yet personable tone
    - 500-800 words optimal length
    
    Write the complete email content now:
    """

def create_generic_content_prompt(deliverable, narrative_theme, briefing, research, state):
    """Create generic content prompt for other content types"""
    return f"""
    Create high-quality content that contributes to our narrative story.
    
    CONTENT DELIVERABLE BRIEF:
    Title: {deliverable['title']}
    Type: {deliverable['content_type']}
    Purpose: {deliverable['content_brief']}
    Key Messages: {', '.join(deliverable['key_messages'])}
    
    NARRATIVE CONTEXT:
    Theme: {narrative_theme}
    Position in Story: {deliverable['narrative_order']} of {state['total_deliverables']}
    
    Create complete, ready-to-use content that serves the specified purpose and contributes to the overall narrative.
    """

def extract_summary(content: str) -> str:
    """Extract a summary from content"""
    lines = content.split('\n')
    # Find first paragraph that's not a header
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and len(line) > 50:
            return line[:200] + "..." if len(line) > 200 else line
    
    # Fallback: use first 200 characters
    clean_content = ' '.join(content.split())
    return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content

# --- Workflow Control Functions ---

def should_create_more_content(state: ContentDeliverableState) -> str:
    """Determine if we need to create more content deliverables"""
    current_index = state['current_deliverable_index']
    total_deliverables = state['total_deliverables']
    
    if current_index < total_deliverables:
        logger.info(f"🔄 Creating deliverable {current_index + 1} of {total_deliverables}")
        return "create_content"
    else:
        logger.info("🎯 All deliverables created, connecting narrative")
        return "connect_narrative"

# --- Graph Construction ---
def create_content_deliverable_workflow():
    """Create the content-first workflow graph"""
    workflow = StateGraph(ContentDeliverableState)
    
    # Add nodes
    workflow.add_node("narrative_coordinator", narrative_coordinator_node)
    workflow.add_node("content_researcher", content_researcher_node) 
    workflow.add_node("content_creator", content_creator_node)
    workflow.add_node("narrative_connector", narrative_connector_node)
    workflow.add_node("content_storage", content_storage_node)
    
    # Set entry point
    workflow.set_entry_point("narrative_coordinator")
    
    # Create workflow edges
    workflow.add_edge("narrative_coordinator", "content_researcher")
    workflow.add_edge("content_researcher", "content_creator")
    
    # Conditional logic: create more content or finish
    workflow.add_conditional_edges(
        "content_creator",
        should_create_more_content,
        {
            "create_content": "content_researcher",  # Loop back for next deliverable
            "connect_narrative": "narrative_connector"
        }
    )
    
    workflow.add_edge("narrative_connector", "content_storage")
    workflow.add_edge("content_storage", END)
    
    return workflow.compile()

# Main workflow app
content_deliverable_app = create_content_deliverable_workflow()

# --- Testing Function ---
def test_content_deliverable_workflow():
    """Test the content-first workflow"""
    print("🚀 Testing Content-First Deliverable Workflow")
    print("=" * 60)
    
    # First create a test campaign
    db_service = DatabaseService()
    
    try:
        # Check if test campaign exists or create one
        test_campaigns = db_service.search_campaigns("test_campaign")
        if test_campaigns:
            campaign_id = test_campaigns[0]['id']
        else:
            # Create a simple test campaign
            campaign_id = db_service.create_campaign({
                'name': 'Content Deliverable Test Campaign',
                'status': 'draft'
            })
        
        print(f"📋 Using test campaign: {campaign_id}")
        
    except Exception as e:
        print(f"⚠️ Could not create test campaign: {e}")
        campaign_id = "00000000-0000-0000-0000-000000000001"  # Fallback UUID
    
    # Example input focusing on deliverables, not tasks
    example_input = {
        "campaign_id": campaign_id,
        "briefing": {
            "marketing_objective": "Position as thought leaders in embedded lending",
            "target_audience": "B2B financial services decision makers",
            "channels": ["blog", "linkedin", "email"],
            "desired_tone": "expert",
            "company_context": "CrediLinq.ai is a fintech leader in embedded lending and B2B credit solutions"
        },
        "content_strategy": {
            "focus": "thought leadership",
            "key_topics": ["embedded lending", "B2B payments", "fintech innovation"]
        }
    }
    
    try:
        result = content_deliverable_app.invoke(example_input)
        
        print("\n📋 CONTENT DELIVERABLES CREATED:")
        print("=" * 60)
        
        for i, deliverable in enumerate(result.get("deliverables", []), 1):
            print(f"\n{i}. {deliverable['title']}")
            print(f"   Type: {deliverable['content_type']}")
            print(f"   Platform: {deliverable['platform']}")
            print(f"   Word Count: {deliverable['word_count']}")
            print(f"   Key Messages: {', '.join(deliverable['key_messages'])}")
            print(f"   Summary: {deliverable['summary']}")
            
            # Show content preview
            content_preview = deliverable['content'][:200] + "..." if len(deliverable['content']) > 200 else deliverable['content']
            print(f"   Preview: {content_preview}")
            
        print(f"\n🎭 NARRATIVE THEME: {result.get('narrative_theme', 'Not specified')}")
        print(f"📚 STORY ARC: {' → '.join(result.get('story_arc', []))}")
        print(f"🔗 CONTENT RELATIONSHIPS: {len(result.get('content_relationships', {}))} connections created")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_content_deliverable_workflow()