#!/usr/bin/env python3
"""
Demo script for multi-format content generation system.
Shows how the content repurposer works with sample data.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Mock the dependencies for demo purposes
class MockChatOpenAI:
    def __init__(self, **kwargs):
        pass
    
    async def agenerate(self, messages):
        # Mock response based on the type of request
        prompt_text = str(messages[0]) if messages else ""
        
        if "LinkedIn" in prompt_text:
            content = """🚀 Just discovered something game-changing about AI in content marketing!

Here's what most companies get wrong: They think AI will replace human creativity. But the real magic happens when AI amplifies human insights.

My experience? AI helps me:
✅ Research faster
✅ Generate more ideas  
✅ Test different angles
✅ Scale personalization

But the strategy, emotion, and authentic voice? That's still 100% human.

What's your experience with AI tools? Are you seeing them as competitors or collaborators?

#AI #ContentMarketing #DigitalStrategy #MarketingTech #Innovation"""
        
        elif "Twitter" in prompt_text:
            content = """1/ 🧵 Thread: The biggest AI misconception in marketing

Most think AI = replacing humans
Reality: AI = amplifying humans 🚀

2/ Here's what AI actually does well:
• Research at lightning speed
• Generate endless variations  
• A/B test everything
• Personalize at scale

3/ What AI can't do:
• Understand your brand soul
• Feel customer pain points
• Make strategic decisions
• Build genuine relationships

4/ The sweet spot? AI handles the grunt work, humans focus on strategy and creativity.

Best combo: Human insight + AI execution

What's your take? Are you team AI-replacement or AI-amplification?

Retweet if this resonates! 🔄"""
        
        elif "Instagram" in prompt_text:
            content = """✨ Let's talk about the AI revolution in marketing... 

I used to spend HOURS on content research and ideation. Now? AI handles the heavy lifting while I focus on what matters most - connecting with YOU! 💫

Here's my game-changing workflow:
🤖 AI helps me research trends
💡 AI generates creative angles  
📊 AI tests different approaches
🎯 But I bring the human touch

The result? More authentic content, better engagement, and time to actually connect with my community! 

Behind the scenes: This post was researched with AI, but every word reflects my genuine experience and passion for helping marketers thrive in the AI age.

What's your biggest question about using AI in your content strategy? Drop it below! 👇

#AIMarketing #ContentStrategy #MarketingTips #DigitalMarketing #AI #ContentCreation #MarketingLife #Entrepreneurship #SocialMedia #Innovation"""
        
        else:
            content = "This is sample generated content for the demo."
        
        class MockGeneration:
            def __init__(self, text):
                self.text = text
        
        class MockGenerations:
            def __init__(self, text):
                self.generations = [[MockGeneration(text)]]
        
        return MockGenerations(content)

# Mock the imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simulate the content repurposer with sample data
class DemoContentRepurposer:
    def __init__(self):
        self.adaptation_llm = MockChatOpenAI()
        self.optimization_llm = MockChatOpenAI()
    
    async def one_click_repurpose(self, blog_content: str, blog_context: Dict[str, Any]):
        """Demo implementation of one-click repurposing."""
        
        # Simulate platform adaptations
        platforms = ["linkedin_post", "twitter_thread", "instagram_post", "facebook_post"]
        repurposed_content = {}
        
        for platform in platforms:
            # Generate platform-specific content
            if platform == "linkedin_post":
                content = """🚀 Just discovered something game-changing about AI in content marketing!

Here's what most companies get wrong: They think AI will replace human creativity. But the real magic happens when AI amplifies human insights.

My experience? AI helps me:
✅ Research faster
✅ Generate more ideas  
✅ Test different angles
✅ Scale personalization

But the strategy, emotion, and authentic voice? That's still 100% human.

What's your experience with AI tools? Are you seeing them as competitors or collaborators?

#AI #ContentMarketing #DigitalStrategy #MarketingTech #Innovation"""
                hashtags = ["AI", "ContentMarketing", "DigitalStrategy", "MarketingTech", "Innovation"]
                cta = "What's your experience with AI tools? Share your thoughts below."
                
            elif platform == "twitter_thread":
                content = """1/ 🧵 The biggest AI misconception in marketing:

Most think AI = replacing humans
Reality: AI = amplifying humans 🚀

2/ Here's what AI actually does well:
• Research at lightning speed  
• Generate endless variations
• A/B test everything
• Personalize at scale

3/ What AI can't do:
• Understand your brand soul
• Feel customer pain points  
• Make strategic decisions
• Build genuine relationships

4/ The sweet spot? 
AI handles grunt work → Humans focus on strategy

Best combo: Human insight + AI execution

What's your take? Team replacement or amplification?

Retweet if this resonates! 🔄"""
                hashtags = ["AI", "Marketing", "Thread", "ContentStrategy", "MarketingTips"]
                cta = "Retweet if this resonates! What's your take?"
                
            elif platform == "instagram_post":
                content = """✨ Let's talk about the AI revolution in marketing...

I used to spend HOURS on content research. Now? AI handles the heavy lifting while I focus on connecting with YOU! 💫

My game-changing workflow:
🤖 AI researches trends
💡 AI generates angles
📊 AI tests approaches  
🎯 I bring human touch

Result? More authentic content, better engagement, and time to connect with my community!

Behind the scenes: This post was researched with AI, but every word reflects my genuine experience.

What's your biggest AI question? Drop it below! 👇

#AIMarketing #ContentStrategy #MarketingTips #DigitalMarketing #ContentCreation"""
                hashtags = ["AIMarketing", "ContentStrategy", "MarketingTips", "DigitalMarketing", "ContentCreation"]
                cta = "What's your biggest AI question? Drop it below!"
                
            else:  # facebook_post
                content = """The AI revolution in marketing is here, and it's not what you think! 🚀

I've been experimenting with AI tools for content creation, and here's what I've learned:

❌ AI won't replace human creativity
✅ AI amplifies human insights

The secret sauce? Using AI for the heavy lifting (research, ideation, testing) while keeping the human touch for strategy and authentic connection.

My results after 3 months:
• 50% less time on research
• 3x more content variations tested
• Higher engagement rates
• More time for genuine community building

The future isn't human vs. AI - it's human + AI working together.

What's your experience with AI in your marketing? I'd love to hear your wins and challenges in the comments! 💬

#AIMarketing #ContentStrategy #MarketingInnovation #DigitalMarketing"""
                hashtags = ["AIMarketing", "ContentStrategy", "MarketingInnovation", "DigitalMarketing"]
                cta = "What's your experience with AI in marketing? Share in the comments!"
            
            repurposed_content[platform] = {
                "content": content,
                "hashtags": hashtags,
                "call_to_action": cta,
                "word_count": len(content.split()),
                "character_count": len(content),
                "estimated_engagement": {
                    "score": 75 + (hash(platform) % 20),  # Random score 75-95
                    "predicted_reach": "high"
                },
                "optimization_notes": [
                    f"Optimized for {platform.replace('_', ' ').title()}",
                    "Good engagement potential",
                    "Platform-appropriate length"
                ]
            }
        
        # Generate content calendar
        content_calendar = {
            "linkedin_post": {
                "suggested_date": "2025-01-15",
                "optimal_time": "09:00",
                "day_of_week": "Wednesday",
                "reasoning": "Optimal timing for LinkedIn professional audience"
            },
            "twitter_thread": {
                "suggested_date": "2025-01-16",
                "optimal_time": "12:00", 
                "day_of_week": "Thursday",
                "reasoning": "Lunch break engagement on Twitter"
            },
            "instagram_post": {
                "suggested_date": "2025-01-17",
                "optimal_time": "18:00",
                "day_of_week": "Friday", 
                "reasoning": "Evening engagement on Instagram"
            },
            "facebook_post": {
                "suggested_date": "2025-01-18",
                "optimal_time": "15:00",
                "day_of_week": "Saturday",
                "reasoning": "Weekend leisure browsing on Facebook"
            }
        }
        
        # Calculate total reach score
        total_reach_score = sum(
            content["estimated_engagement"]["score"] 
            for content in repurposed_content.values()
        ) / len(repurposed_content)
        
        # Generate recommendations
        recommendations = [
            "LinkedIn content shows highest engagement potential - prioritize this platform",
            "Twitter thread format works well for educational content - consider more threads",
            "Instagram posts benefit from visual elements - prepare supporting graphics", 
            "Facebook posts perform better with personal stories - add more narrative",
            "Schedule posts across 4 days to maximize reach without audience fatigue",
            "Monitor first 24-hour performance to optimize future content"
        ]
        
        return {
            "repurposed_content": repurposed_content,
            "content_calendar": content_calendar,
            "performance_predictions": {
                platform: {
                    "estimated_reach": 150 + (hash(platform) % 100),
                    "estimated_engagement_rate": f"{content['estimated_engagement']['score']/10:.1f}%",
                    "predicted_interactions": 15 + (hash(platform) % 20),
                    "confidence_level": "high"
                }
                for platform, content in repurposed_content.items()
            },
            "total_reach_score": round(total_reach_score),
            "recommendations": recommendations,
            "created_at": datetime.now().isoformat()
        }

async def demo_content_repurposing():
    """Run a demo of the content repurposing system."""
    
    print("🎯 CrediLinQ Multi-Format Content Generation Demo")
    print("=" * 55)
    print()
    
    # Sample blog content
    blog_content = """
    The Future of AI-Powered Content Marketing: A Strategic Guide
    
    Artificial Intelligence is revolutionizing content marketing, but not in the way most people think. 
    Instead of replacing human creativity, AI is amplifying it, enabling marketers to produce more 
    personalized, data-driven content at scale.
    
    Key insights from recent industry research:
    
    1. AI excels at data analysis and pattern recognition
    2. Human creativity and strategy remain irreplaceable  
    3. The best results come from human-AI collaboration
    4. Personalization at scale is now achievable
    5. Content testing and optimization can be automated
    
    For businesses looking to leverage AI in their content strategy, the focus should be on:
    - Using AI for research and ideation
    - Automating repetitive tasks
    - Testing content variations
    - Personalizing content for different segments
    - Maintaining human oversight for strategy and brand voice
    
    The future belongs to marketers who can effectively combine human insight with AI capabilities.
    """
    
    blog_context = {
        "company_context": "CrediLinQ - AI-powered business solutions platform",
        "target_audience": "marketing professionals and business owners",
        "tone_preference": "professional",
        "include_hashtags": True,
        "include_cta": True
    }
    
    print("📝 Original Blog Content:")
    print("-" * 30)
    print(blog_content.strip()[:200] + "..." if len(blog_content) > 200 else blog_content.strip())
    print()
    
    # Initialize demo repurposer
    repurposer = DemoContentRepurposer()
    
    print("🔄 Processing one-click repurposing...")
    print()
    
    # Perform repurposing
    result = await repurposer.one_click_repurpose(blog_content, blog_context)
    
    # Display results
    print("✅ REPURPOSING COMPLETE!")
    print("=" * 25)
    print()
    
    print(f"📊 Overall Performance Score: {result['total_reach_score']}/100")
    print(f"📅 Content Calendar: {len(result['content_calendar'])} posts scheduled")
    print(f"🎯 Total Reach Potential: {sum(p['estimated_reach'] for p in result['performance_predictions'].values())} impressions")
    print()
    
    # Show repurposed content for each platform
    for platform, content_data in result['repurposed_content'].items():
        platform_name = platform.replace('_', ' ').title()
        print(f"📱 {platform_name.upper()}")
        print("-" * (len(platform_name) + 4))
        
        print(f"📝 Content ({content_data['character_count']} chars):")
        preview = content_data['content'][:300] + "..." if len(content_data['content']) > 300 else content_data['content']
        print(preview)
        print()
        
        print(f"🏷️  Hashtags: {', '.join(['#' + tag for tag in content_data['hashtags'][:3]])}...")
        print(f"📈 Engagement Score: {content_data['estimated_engagement']['score']}/100")
        print(f"📅 Scheduled: {result['content_calendar'][platform]['suggested_date']} at {result['content_calendar'][platform]['optimal_time']}")
        print()
    
    # Show recommendations
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 35)
    for i, rec in enumerate(result['recommendations'][:4], 1):
        print(f"{i}. {rec}")
    print()
    
    # Show performance predictions
    print("📊 PERFORMANCE PREDICTIONS")  
    print("-" * 28)
    for platform, pred in result['performance_predictions'].items():
        platform_name = platform.replace('_', ' ').title()
        print(f"{platform_name:15} | Reach: {pred['estimated_reach']:3d} | Engagement: {pred['estimated_engagement_rate']:5s} | Interactions: {pred['predicted_interactions']:2d}")
    
    print()
    print("🎉 Demo Complete! The multi-format content generation system successfully:")
    print("   ✅ Adapted content for 4 different platforms")
    print("   ✅ Optimized length and tone for each platform")  
    print("   ✅ Generated platform-specific hashtags and CTAs")
    print("   ✅ Created an optimized content calendar")
    print("   ✅ Predicted performance metrics")
    print("   ✅ Provided actionable recommendations")

if __name__ == "__main__":
    asyncio.run(demo_content_repurposing())