# agent.py (Refactored for Multi-Agent workflow with Gemini and RAG)
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

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Expanded State for the Multi-Agent Workflow ---
class BlogWriterState(TypedDict):
    blog_title: str
    company_context: str
    content_type: str  # "linkedin" or "blog"
    outline: Annotated[List[str], "The blog post's outline"]
    research: Annotated[dict, "Research keyed by section title"]
    draft: Annotated[str, "The current draft of the blog post"]
    review_notes: Annotated[str, "Notes from the editor for revision"]
    final_post: str

# --- 2. Shared Tools for the Agent Team ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- 3. The Agent Team: Node Definitions ---
def planner_node(state: BlogWriterState):
    """Creates a detailed outline for the blog post"""
    print("🎯 --- AGENT: Planner ---")
    
    content_type = state.get('content_type', 'blog').lower()
    
    if content_type == 'linkedin':
        prompt = f"""You are an expert LinkedIn content strategist. Create a detailed outline for a LinkedIn post.

Title: {state['blog_title']}
Company Context: {state['company_context']}
Content Type: LinkedIn Post

Create a logical, engaging outline optimized for LinkedIn that will result in a professional, engaging post.
LinkedIn posts should be:
- Concise and impactful (800-1200 words max)
- Include a strong hook in the opening
- Have clear value propositions
- Include actionable insights
- End with a call-to-action or question for engagement

Your output must be a Python list of strings representing section titles.

Example format for LinkedIn: ["Hook & Opening Statement", "Key Insight #1", "Key Insight #2", "Real-world Example", "Call to Action"]

Return ONLY the Python list, nothing else."""
    else:
        prompt = f"""You are an expert content strategist. Create a detailed outline for a comprehensive blog post.

Title: {state['blog_title']}
Company Context: {state['company_context']}
Content Type: Blog Post

Create a logical, engaging outline that will result in a comprehensive, in-depth blog post.
Blog posts should be:
- Detailed and educational (1500-2500 words)
- Include comprehensive coverage of the topic
- Have multiple sections with deep insights
- Include practical implementation details
- Provide thorough analysis and examples

Your output must be a Python list of strings representing section titles.

Example format for Blog: ["Introduction", "Why X is Important", "How to Implement X", "Best Practices", "Common Pitfalls", "Case Studies", "Conclusion"]

Return ONLY the Python list, nothing else."""

    response = llm.invoke(prompt)
    try:
        outline = ast.literal_eval(response.content.strip())
        print(f"📝 Created {content_type} outline with {len(outline)} sections")
        return {"outline": outline}
    except:
        # Fallback if parsing fails
        if content_type == 'linkedin':
            outline = ["Hook & Opening", "Key Insight", "Value Proposition", "Call to Action"]
        else:
            outline = ["Introduction", "Main Content", "Key Benefits", "Implementation", "Conclusion"]
        print(f"⚠️  Used fallback {content_type} outline due to parsing error")
        return {"outline": outline}

def researcher_node(state: BlogWriterState):
    """
    For each section in the outline, performs a similarity search against the document_chunks table in Supabase (pgvector).
    Uses OpenAIEmbeddings to embed the query and retrieves the most similar chunks.
    """
    print("🔍 --- AGENT: Researcher (Supabase pgvector) ---")
    research_results = {}
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor()
    for section in state['outline']:
        query = f"Find relevant information for a blog section titled '{section}' on the main topic of '{state['blog_title']}'."
        print(f"  🔎 Researching: {section}")
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
        else:
            research_results[section] = f"No specific research found for {section}"
    cur.close()
    conn.close()
    print(f"📚 Research completed for {len(research_results)} sections")
    return {"research": research_results}

def writer_node(state: BlogWriterState):
    """Writes the blog post draft using the outline and research"""
    print("✍️  --- AGENT: Writer ---")
    research_text = "\n\n".join([f"## Section: {sec}\nResearch: {res}" for sec, res in state['research'].items()])
    review_notes = ""
    if state.get('review_notes'):
        review_notes = f"\n\nIMPORTANT - EDITOR'S REVISION NOTES:\n{state['review_notes']}\nPlease incorporate these revisions in your rewrite."
    
    content_type = state.get('content_type', 'blog').lower()
    
    if content_type == 'linkedin':
        prompt = f"""You are 'ContextMark', an expert LinkedIn content creator with 15+ years of experience in professional social media. Write an engaging, professional LinkedIn post.

LINKEDIN POST TITLE: {state['blog_title']}

OUTLINE TO FOLLOW:
{state['outline']}

RESEARCH TO USE:
---
{research_text}
---

COMPANY CONTEXT & TONE:
{state['company_context']}

{review_notes}

LINKEDIN-SPECIFIC REQUIREMENTS:
- Write in a professional yet personable tone that encourages engagement
- Use the research as your primary source of truth
- Follow the outline structure closely
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
        prompt = f"""You are 'ContextMark', an expert blog writer with 15+ years of experience. Write a comprehensive, engaging blog post.

BLOG TITLE: {state['blog_title']}

OUTLINE TO FOLLOW:
{state['outline']}

RESEARCH TO USE:
---
{research_text}
---

COMPANY CONTEXT & TONE:
{state['company_context']}

{review_notes}

BLOG-SPECIFIC REQUIREMENTS:
- Write in professional yet conversational tone
- Use the research as your primary source of truth
- Follow the outline structure closely
- Include engaging introduction and strong conclusion
- Use proper Markdown formatting with headers (## for main sections)
- Aim for 1500-2500 words for comprehensive coverage
- Include actionable insights and practical advice
- Provide detailed explanations and examples
- Use bullet points and numbered lists where appropriate
- Maintain consistency with company voice

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
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
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
        print("1. Set OPENAI_API_KEY and SUPABASE_DB_URL in your .env file")
        print("2. Use process_and_embed_document to load your knowledge base") 