#!/usr/bin/env python3
"""
LangGraph Workflows for CrediLinq Campaign Content Generation
This file contains the main workflows that LangGraph Studio can monitor.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
# No checkpointer imports needed for basic functionality

@dataclass
class CampaignState:
    """State for campaign content generation workflow"""
    campaign_id: str
    campaign_name: str = ""
    target_audience: str = ""
    content_types: List[str] = None
    key_topics: List[str] = None
    generated_content: Dict[str, Any] = None
    agent_outputs: Dict[str, Any] = None
    current_step: str = "planning"
    status: str = "running"
    errors: List[str] = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = []
        if self.generated_content is None:
            self.generated_content = {}
        if self.agent_outputs is None:
            self.agent_outputs = {}
        if self.errors is None:
            self.errors = []


def planner_agent(state: CampaignState) -> CampaignState:
    """Planning agent for content strategy"""
    print(f"🎯 Planning content for campaign: {state.campaign_id}")
    
    # Simulate planning logic
    state.agent_outputs["planner"] = {
        "content_strategy": f"Comprehensive fintech content strategy for {state.campaign_name}",
        "key_themes": state.key_topics or ["fintech innovation", "digital transformation"],
        "content_calendar": "Q4 2025 content calendar created",
        "timestamp": datetime.now().isoformat()
    }
    
    state.current_step = "research"
    return state


def researcher_agent(state: CampaignState) -> CampaignState:
    """Research agent for market analysis"""
    print(f"🔍 Researching market trends for campaign: {state.campaign_id}")
    
    state.agent_outputs["researcher"] = {
        "market_trends": ["AI in fintech", "Embedded finance", "Open banking"],
        "competitor_analysis": "Top 10 fintech competitors analyzed",
        "industry_insights": "Q3 2025 fintech industry report findings",
        "timestamp": datetime.now().isoformat()
    }
    
    state.current_step = "content_creation"
    return state


def writer_agent(state: CampaignState) -> CampaignState:
    """Content writing agent"""
    print(f"✍️ Creating content for campaign: {state.campaign_id}")
    
    content_pieces = []
    for content_type in state.content_types:
        if content_type == "blog_posts":
            content_pieces.append({
                "type": "blog_post",
                "title": f"The Future of Fintech: Insights from {state.campaign_name}",
                "content": "Comprehensive blog post about fintech trends...",
                "word_count": 1500,
                "seo_keywords": ["fintech", "digital banking", "financial innovation"]
            })
        elif content_type == "social_media":
            content_pieces.append({
                "type": "social_media_post",
                "platform": "LinkedIn",
                "content": f"Exciting insights from {state.campaign_name}! 🚀 #Fintech #Innovation",
                "engagement_hooks": ["Question", "Call-to-action", "Industry insights"]
            })
        elif content_type == "email_newsletter":
            content_pieces.append({
                "type": "email_newsletter",
                "subject": f"Key Takeaways from {state.campaign_name}",
                "content": "Weekly fintech newsletter with curated insights...",
                "segments": ["executives", "developers", "investors"]
            })
    
    state.generated_content["content_pieces"] = content_pieces
    state.agent_outputs["writer"] = {
        "content_count": len(content_pieces),
        "total_words": sum(piece.get("word_count", 500) for piece in content_pieces),
        "content_types": list(set(piece["type"] for piece in content_pieces)),
        "timestamp": datetime.now().isoformat()
    }
    
    state.current_step = "editing"
    return state


def editor_agent(state: CampaignState) -> CampaignState:
    """Content editing and quality assurance agent"""
    print(f"📝 Editing content for campaign: {state.campaign_id}")
    
    # Simulate editing process
    if "content_pieces" in state.generated_content:
        for piece in state.generated_content["content_pieces"]:
            piece["edited"] = True
            piece["quality_score"] = 0.85
            piece["readability_score"] = 0.90
    
    state.agent_outputs["editor"] = {
        "pieces_edited": len(state.generated_content.get("content_pieces", [])),
        "average_quality": 0.85,
        "improvements": ["Grammar check", "SEO optimization", "Brand compliance"],
        "timestamp": datetime.now().isoformat()
    }
    
    state.current_step = "seo_optimization"
    return state


def seo_agent(state: CampaignState) -> CampaignState:
    """SEO optimization agent"""
    print(f"🔍 Optimizing SEO for campaign: {state.campaign_id}")
    
    state.agent_outputs["seo"] = {
        "keywords_optimized": 25,
        "meta_descriptions": "Generated for all content pieces",
        "internal_links": "Cross-linking strategy implemented",
        "seo_score": 0.88,
        "timestamp": datetime.now().isoformat()
    }
    
    state.current_step = "finalization"
    return state


def should_continue(state: CampaignState) -> str:
    """Conditional logic for workflow routing"""
    step_mapping = {
        "research": "content_creation", 
        "content_creation": "editing",
        "editing": "seo_optimization",
        "seo_optimization": END
    }
    
    return step_mapping.get(state.current_step, END)


def create_campaign_workflow():
    """Create the main campaign content generation workflow"""
    
    # Initialize the StateGraph
    workflow = StateGraph(CampaignState)
    
    # Add nodes (agents)
    workflow.add_node("planning", planner_agent)
    workflow.add_node("research", researcher_agent)  
    workflow.add_node("content_creation", writer_agent)
    workflow.add_node("editing", editor_agent)
    workflow.add_node("seo_optimization", seo_agent)
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Add direct edges for simple workflow
    workflow.add_edge("planning", "research")
    workflow.add_edge("research", "content_creation")
    workflow.add_edge("content_creation", "editing")
    workflow.add_edge("editing", "seo_optimization")
    workflow.add_edge("seo_optimization", END)
    
    # Compile workflow without checkpointer (for LangGraph Studio visibility)
    return workflow.compile()


# Create the main workflow that LangGraph Studio can detect
campaign_workflow = create_campaign_workflow()


def run_campaign_workflow(campaign_id: str, campaign_name: str = "Money2020", 
                         target_audience: str = "fintech professionals",
                         content_types: List[str] = None,
                         key_topics: List[str] = None):
    """Run the campaign workflow with given parameters"""
    
    if content_types is None:
        content_types = ["blog_posts", "social_media", "email_newsletter"]
    
    if key_topics is None:
        key_topics = ["fintech innovation", "digital banking", "payment solutions"]
    
    initial_state = CampaignState(
        campaign_id=campaign_id,
        campaign_name=campaign_name,
        target_audience=target_audience,
        content_types=content_types,
        key_topics=key_topics
    )
    
    config = {"configurable": {"thread_id": campaign_id}}
    
    try:
        # Run the workflow
        result = campaign_workflow.invoke(initial_state, config=config)
        return result
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    result = run_campaign_workflow(
        campaign_id="94c61288-b2c1-427b-8004-5c659489a251",
        campaign_name="Money2020",
        target_audience="fintech professionals and executives",
        content_types=["blog_posts", "social_media", "email_newsletter"],
        key_topics=["fintech innovation", "digital banking", "payment solutions", "financial inclusion"]
    )
    
    if result:
        print("✅ Workflow completed successfully!")
        print(f"Final state: {result.get('status', 'completed')}")
        print(f"Generated content pieces: {len(result.get('generated_content', {}).get('content_pieces', []))}")
        print(f"Current step: {result.get('current_step', 'finalization')}")
    else:
        print("❌ Workflow execution failed")