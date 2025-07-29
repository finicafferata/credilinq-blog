# Blog workflow agent 
import os
from typing import TypedDict, List
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import ast
import psycopg2
import numpy as np
import time
import logging

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/credilinq_dev_postgres?sslmode=disable")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_to_supabase_with_retry(max_retries=3, initial_wait=1):
    """
    Conecta a Supabase con reintentos automáticos para manejar problemas de red temporales
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Intento de conexión a Supabase: {attempt + 1}/{max_retries}")
            conn = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=30,  # More generous timeout
                sslmode='disable'    # Disable SSL for local development
            )
            conn.autocommit = True
            logger.info("✅ Conexión a Supabase exitosa")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"❌ Intento {attempt + 1} falló: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.info(f"⏳ Esperando {wait_time}s antes del siguiente intento...")
                time.sleep(wait_time)
            else:
                logger.error(f"🚨 All connection attempts failed after {max_retries} attempts")
                raise e
        except Exception as e:
            logger.error(f"❌ Error inesperado en conexión: {str(e)}")
            raise e

# --- 1. Expanded State for the Multi-Agent Workflow ---
class BlogWriterState(TypedDict):
    blog_title: str
    company_context: str
    content_type: str  # "linkedin" or "blog"
    outline: Annotated[List[str], "The blog post's outline"]
    research: Annotated[dict, "Research keyed by section title"]
    geo_metadata: Annotated[dict, "GEO optimization package"]
    draft: Annotated[str, "The current draft of the blog post"]
    review_notes: Annotated[str, "Notes from the editor for revision"]
    final_post: str

# --- 2. Shared Tools for the Agent Team ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- 3. Node Functions ---

def fallback_local_research(state: BlogWriterState):
    """
    Fallback function that uses local files when Supabase is not available
    """
    logger.info("🔄 Usando investigación local como fallback")
    research_results = {}
    
    # Search for files in knowledge_base
    knowledge_base_path = "knowledge_base"
    if os.path.exists(knowledge_base_path):
        for section in state['outline']:
            print(f"  🔎 Researching locally: {section}")
            section_research = []
            
            # Read available files
            for filename in os.listdir(knowledge_base_path):
                if filename.endswith('.txt'):
                    try:
                        file_path = os.path.join(knowledge_base_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Search for relevant content (simple keyword matching)
                            keywords = section.lower().split()
                            if any(keyword in content.lower() for keyword in keywords):
                                # Take the first 500 characters as a sample
                                sample = content[:500] + "..." if len(content) > 500 else content
                                section_research.append(sample)
                    except Exception as e:
                        logger.warning(f"Error leyendo {filename}: {str(e)}")
                        continue
            
            # Combine research for this section
            if section_research:
                research_results[section] = "\n\n".join(section_research)
            else:
                research_results[section] = f"Información general sobre {section} basada en contexto de {state['company_context']}"
    else:
        logger.warning("❌ Directorio knowledge_base no encontrado")
        # Final fallback: minimal research
        for section in state['outline']:
            research_results[section] = f"Información general sobre {section} basada en contexto de {state['company_context']}"
    
    return {**state, "research": research_results}

def outliner_node(state: BlogWriterState):
    """Creates a detailed outline for the blog post"""
    print("📝 --- AGENT: Outliner ---")
    # Create a structured outline prompt based on content type
    content_format = "LinkedIn post format" if state["content_type"] == "linkedin" else "blog post format"
    prompt = f"""
    Create a detailed outline for a {content_format} about: {state['blog_title']}
    Company context: {state['company_context']}
    
    Return ONLY a Python list of section titles, like:
    ["Introduction", "Main Topic", "Key Benefits", "Conclusion"]
    
    For LinkedIn posts, keep sections short and engaging.
    For blog posts, create more comprehensive sections.
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    try:
        # Parse the outline from the response
        outline = ast.literal_eval(response.content.strip())
        print(f"📋 Created outline with {len(outline)} sections: {outline}")
        return {**state, "outline": outline}
    except:
        # Fallback outline if parsing fails
        fallback_outline = ["Introduction", "Main Content", "Conclusion"]
        print(f"📋 Using fallback outline: {fallback_outline}")
        return {**state, "outline": fallback_outline}

def geo_optimizer_node(state: BlogWriterState):
    """Generates GEO optimization metadata for the content"""
    print("🎯 --- AGENT: GEO Optimizer ---")
    
    try:
        from ..specialized.geo_agent import GEOAgent
        
        geo_agent = GEOAgent()
        
        # Prepare input data for GEO agent
        geo_input = {
            "topic": state['blog_title'],
            "target_audience": "Business professionals and industry experts",  # Default, could be parameterized
            "unique_angle": f"Expert insights from {state['company_context']}",
            "author_brand": state['company_context'].split()[0] if state['company_context'] else "Industry Expert",
            "company_context": state['company_context']
        }
        
        result = geo_agent.execute(geo_input)
        
        if result.success:
            geo_metadata = result.data.get('geo_strategy', {})
            print(f"📊 GEO optimization completed with score: {result.data.get('optimization_score', 0)}")
            return {**state, "geo_metadata": geo_metadata}
        else:
            print(f"⚠️ GEO optimization failed: {result.error_message}")
            # Fallback: empty GEO metadata
            return {**state, "geo_metadata": {}}
            
    except Exception as e:
        print(f"❌ Error in GEO optimization: {str(e)}")
        # Fallback: empty GEO metadata
        return {**state, "geo_metadata": {}}

def researcher_node(state: BlogWriterState):
    """
    For each section in the outline, performs a similarity search against the document_chunks table in Supabase (pgvector).
    Uses OpenAIEmbeddings to embed the query and retrieves the most similar chunks.
    """
    print("🔍 --- AGENT: Researcher (Supabase pgvector) ---")
    research_results = {}
    
    try:
        # Use robust connection with retries
        conn = connect_to_supabase_with_retry()
        cur = conn.cursor()
    except Exception as e:
        logger.error(f"🚨 No se pudo conectar a Supabase: {str(e)}")
        print("⚠️ Fallback: Usando conocimiento base local")
        return fallback_local_research(state)
    try:
        for section in state['outline']:
            query = f"Find relevant information for a blog section titled '{section}' on the main topic of '{state['blog_title']}'."
            print(f"  🔎 Researching: {section}")
            
            try:
                # Generate embedding for the query
                query_embedding = embeddings_model.embed_query(query)
                # Prepare embedding for SQL (as array)
                embedding_str = '[' + ','.join([str(x) for x in query_embedding]) + ']'
                
                # SQL: Find top 3 most similar chunks using cosine distance
                cur.execute(
                    """
                    SELECT content, (embedding <#> %s::vector) AS distance
                    FROM document_chunks
                    ORDER BY distance ASC
                    LIMIT 3;
                    """,
                    (embedding_str,)
                )
                rows = cur.fetchall()
                if rows:
                    research_content = "\n\n".join([row[0] for row in rows])
                    research_results[section] = research_content
                    print(f"    📄 Found {len(rows)} relevant chunks")
                else:
                    research_results[section] = f"No specific research found for {section}"
                    print(f"    ⚠️ No research found for {section}")
            except Exception as e:
                logger.warning(f"Error investigando sección '{section}': {str(e)}")
                # Fallback for this specific section
                research_results[section] = f"Información general sobre {section} basada en {state['company_context']}"
                
    except Exception as e:
        logger.error(f"Error durante la investigación: {str(e)}")
        research_results = {section: f"Error de investigación para {section}" for section in state['outline']}
    finally:
        # Close connection safely
        try:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
                logger.info("🔒 Conexión cerrada de forma segura")
        except Exception as e:
            logger.warning(f"Error cerrando conexión: {str(e)}")
    
    print(f"📚 Research completed for {len(research_results)} sections")
    return {**state, "research": research_results}

def writer_node(state: BlogWriterState):
    """Writes the blog post draft using the outline, research, and GEO metadata"""
    print("✍️  --- AGENT: Writer ---")
    research_text = "\n\n".join([f"## Section: {sec}\nResearch: {res}" for sec, res in state['research'].items()])
    
    # Include GEO metadata in the writing process
    geo_guidance = ""
    if state.get('geo_metadata'):
        geo_meta = state['geo_metadata']
        geo_guidance = f"""
GEO OPTIMIZATION GUIDANCE:
- Optimized Title: {geo_meta.get('optimized_title', '')}
- Key Takeaways to Include: {geo_meta.get('key_takeaways', [])}
- FAQ Questions to Address: {[faq.get('question', '') for faq in geo_meta.get('faq_for_structured_data', [])]}
- Important Concepts: {list(geo_meta.get('entity_and_concept_map', {}).keys())}
- Expert Quote to Incorporate: "{geo_meta.get('citable_expert_quote', {}).get('quote', '')}"
"""
    
    review_notes = ""
    if state.get('review_notes'):
        review_notes = f"\n\nIMPORTANT - EDITOR'S REVISION NOTES:\n{state['review_notes']}\nPlease incorporate these revisions in your rewrite."
    
    content_type = state.get('content_type', 'blog').lower()
    
    if content_type == 'linkedin':
        prompt = f"""You are 'ContextMark', an expert LinkedIn content creator with 15+ years of experience in professional social media. Write an engaging, professional LinkedIn post optimized for AI Answer Engines.

LINKEDIN POST TITLE: {state['blog_title']}

OUTLINE TO FOLLOW:
{state['outline']}

RESEARCH TO USE:
---
{research_text}
---

{geo_guidance}

COMPANY CONTEXT & TONE:
{state['company_context']}

{review_notes}

LINKEDIN-SPECIFIC REQUIREMENTS:
- Write in a professional yet personable tone that encourages engagement
- Use the research as your primary source of truth
- Follow the outline structure closely
- Incorporate GEO guidance for AI discoverability
- Start with a compelling hook that grabs attention
- Include relevant emojis sparingly and professionally
- Aim for 800-1200 words (LinkedIn optimal length)
- Include actionable insights that professionals can implement
- End with a call-to-action or thought-provoking question
- Use line breaks and short paragraphs for mobile readability
- Include relevant hashtags at the end (3-5 maximum)
- Maintain consistency with company voice and professional brand

Write the complete LinkedIn post now. Use minimal formatting - just line breaks and emojis where appropriate."""
    else:
        prompt = f"""You are 'ContextMark', an expert blog writer with 15+ years of experience. Write a comprehensive, engaging blog post optimized for AI Answer Engines.

BLOG TITLE: {state['blog_title']}

OUTLINE TO FOLLOW:
{state['outline']}

RESEARCH TO USE:
---
{research_text}
---

{geo_guidance}

COMPANY CONTEXT & TONE:
{state['company_context']}

{review_notes}

BLOG-SPECIFIC REQUIREMENTS:
- Write in professional yet conversational tone
- Use the research as your primary source of truth
- Follow the outline structure closely
- Incorporate GEO guidance for maximum AI discoverability
- Include engaging introduction and strong conclusion
- Use proper Markdown formatting with headers (## for main sections)
- Aim for 1500-2500 words for comprehensive coverage
- Include actionable insights and practical advice
- Provide detailed explanations and examples
- Use bullet points and numbered lists where appropriate
- Maintain consistency with company voice
- Structure content to answer the FAQ questions from GEO guidance
- Include the expert quote naturally in the content

Write the complete blog post in Markdown format now."""
    
    response = llm.invoke(prompt)
    print(f"📄 {content_type.title()} draft completed ({len(response.content)} characters)")
    return {"draft": response.content}

def editor_node(state: BlogWriterState):
    """Reviews the draft and either approves it or requests revisions"""
    print("🔍 --- AGENT: Editor ---")
    prompt = f"""You are a senior editor reviewing a blog post draft. Evaluate it for quality, clarity, and alignment with company standards.

COMPANY CONTEXT:
{state['company_context']}

BLOG TITLE: {state['blog_title']}

DRAFT TO REVIEW:
---
{state['draft']}
---

EVALUATION CRITERIA:
- Content quality and accuracy
- Alignment with company voice and context
- Proper structure and flow
- Engagement and readability
- Actionable value for readers
- Professional presentation

INSTRUCTIONS:
If the draft meets high standards and is ready for publication, respond with EXACTLY: "APPROVED"

If revisions are needed, provide specific, actionable feedback in bullet points for what the writer should improve. Be constructive and specific."""
    response = llm.invoke(prompt)
    if "APPROVED" in response.content.upper():
        print("✅ --- Editor approved the draft ---")
        return {"review_notes": None, "final_post": state['draft']}
    else:
        print("📝 --- Editor requested revisions ---")
        return {"review_notes": response.content}

# --- 4. Conditional Logic for the Review Loop ---
def should_continue(state: BlogWriterState) -> str:
    """Determines whether to continue with revisions or end the workflow"""
    if state.get("review_notes"):
        print("🔄 Sending back to writer for revisions...")
        return "writer"  # Revisions needed, go back to the writer
    else:
        print("🎉 Workflow complete - blog post approved!")
        return END  # Approved, end the workflow

# --- 5. Graph Construction ---
workflow = StateGraph(BlogWriterState)
workflow.add_node("outliner", outliner_node)
workflow.add_node("geo_optimizer", geo_optimizer_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.set_entry_point("outliner")
workflow.add_edge("outliner", "geo_optimizer")
workflow.add_edge("geo_optimizer", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")
workflow.add_conditional_edges(
    "editor", 
    should_continue, 
    {"writer": "writer", END: END}
)
app = workflow.compile()

if __name__ == "__main__":
    print("🚀 Testing Multi-Agent Content Writing System")
    print("=" * 60)
    example_input = {
        "blog_title": "How Embedded Lending is Transforming B2B Payments",
        "company_context": "Credilinq.ai is a fintech leader in embedded lending and B2B credit solutions across Southeast Asia.",
        "content_type": "blog"  # or "linkedin"
    }
    try:
        result = app.invoke(example_input)
        print("\n📋 FINAL RESULT:")
        print("=" * 60)
        print(result.get("final_post", "No final post generated"))
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY and DATABASE_URL in your .env file")
        print("2. Use process_and_embed_document to load your knowledge base") 