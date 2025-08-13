#!/usr/bin/env python3
"""
Script to populate sample data for testing the analytics dashboard.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import uuid
import random

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.database import db_config

def create_sample_blogs():
    """Create sample blog posts."""
    sample_blogs = []
    titles = [
        "AI Revolution in Content Marketing",
        "The Future of Marketing Automation",
        "Building Better Customer Experiences",
        "Data-Driven Marketing Strategies",
        "Content Creation with AI Agents"
    ]
    
    for i, title in enumerate(titles):
        blog_id = str(uuid.uuid4())
        created_at = (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        
        blog = {
            "id": blog_id,
            "title": title,
            "content_markdown": f"# {title}\n\nSample content for {title}...",
            "status": "published",
            "created_at": created_at,
            "updated_at": created_at,
            "word_count": random.randint(800, 2000),
            "reading_time": random.randint(3, 8)
        }
        sample_blogs.append(blog)
    
    return sample_blogs

def create_sample_campaigns():
    """Create sample campaigns."""
    sample_campaigns = []
    campaign_names = [
        "Q4 Content Push",
        "Product Launch Campaign",
        "Brand Awareness Drive",
        "Customer Retention Focus"
    ]
    
    for i, name in enumerate(campaign_names):
        campaign_id = str(uuid.uuid4())
        created_at = (datetime.now() - timedelta(days=random.randint(1, 45))).isoformat()
        
        campaign = {
            "id": campaign_id,
            "name": name,
            "status": random.choice(["draft", "active", "completed"]),
            "created_at": created_at,
            "updated_at": created_at
        }
        sample_campaigns.append(campaign)
    
    return sample_campaigns

def create_sample_agent_performance():
    """Create sample agent performance data."""
    sample_performance = []
    agent_types = ["planner", "researcher", "writer", "editor", "seo", "image_prompt_generator"]
    
    for _ in range(50):  # Create 50 sample records
        perf_id = str(uuid.uuid4())
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        agent_type = random.choice(agent_types)
        
        start_time = datetime.now() - timedelta(
            days=random.randint(1, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        duration = random.randint(1000, 30000)  # 1-30 seconds
        end_time = start_time + timedelta(milliseconds=duration)
        
        input_tokens = random.randint(100, 2000)
        output_tokens = random.randint(50, 1000)
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * random.uniform(0.001, 0.003)  # Rough cost estimate
        
        performance = {
            "id": perf_id,
            "agent_name": f"{agent_type}_agent",
            "agent_type": agent_type,
            "execution_id": execution_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "status": random.choice(["success", "success", "success", "error"]),  # 75% success
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "created_at": start_time.isoformat()
        }
        sample_performance.append(performance)
    
    return sample_performance

def populate_data():
    """Populate the database with sample data."""
    try:
        # Use the database service instead
        from src.agents.core.database_service import get_db_service
        db_service = get_db_service()
        
        if not db_service.use_supabase or not db_service.supabase:
            print("⚠️  Supabase not available. Cannot populate sample data.")
            return
            
        supabase = db_service.supabase
        
        print("Creating sample blog posts...")
        blogs = create_sample_blogs()
        for blog in blogs:
            try:
                result = supabase.table("blog_posts").insert(blog).execute()
                print(f"Created blog: {blog['title']}")
            except Exception as e:
                print(f"Error creating blog {blog['title']}: {e}")
        
        print("Creating sample campaigns...")
        campaigns = create_sample_campaigns()
        for campaign in campaigns:
            try:
                result = supabase.table("campaigns").insert(campaign).execute()
                print(f"Created campaign: {campaign['name']}")
            except Exception as e:
                print(f"Error creating campaign {campaign['name']}: {e}")
        
        print("Creating sample agent performance data...")
        performances = create_sample_agent_performance()
        for perf in performances:
            try:
                result = supabase.table("agent_performance").insert(perf).execute()
                print(f"Created performance record for {perf['agent_type']}")
            except Exception as e:
                print(f"Error creating performance record: {e}")
        
        print("Sample data population completed!")
        
    except Exception as e:
        print(f"Error populating data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    populate_data()