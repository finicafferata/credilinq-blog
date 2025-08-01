"""
SEO Agent - Analyzes and optimizes content for search engines
"""

# To run this agent, ensure you have the following dependencies installed:
# pip install openai textstat

import os
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import openai
from collections import Counter

# It's recommended to use a robust text analysis library like textstat
try:
    import textstat
except ImportError:
    textstat = None
    logging.warning("`textstat` library not found. Readability and syllable counting will be less accurate. Please run 'pip install textstat'.")

from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError
from ...config import get_settings

logger = logging.getLogger(__name__)

class SEOAgent(BaseAgent):
    """
    Agent specialized in SEO analysis and optimization.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "SEOAgent", description: str = "Advanced AI-powered SEO analysis and optimization"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.SEO,
                name=name,
                description=description,
                capabilities=[
                    "keyword_analysis",
                    "semantic_analysis", 
                    "competitor_analysis",
                    "technical_seo",
                    "serp_analysis",
                    "content_optimization",
                    "schema_markup",
                    "topical_authority"
                ]
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.SEO
        self.settings = get_settings()
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if self.settings.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Analyze and optimize content for SEO.
        
        Args:
            context: Dictionary containing:
                - content: Blog content
                - blog_title: Title of the blog
                - outline: Blog outline sections
                - target_keywords: Target keywords (optional)
        
        Returns:
            AgentResult with SEO analysis and recommendations
        """
        try:
            if not self.validate_input(context):
                raise AgentExecutionError("SEOAgent", "validation", "Invalid input provided. 'content' and 'blog_title' are required.")
                
            logger.info(f"SEOAgent executing for blog: {context.get('blog_title', 'Unknown')}")
            
            content = context.get('content', '')
            blog_title = context.get('blog_title', '')
            outline = context.get('outline', [])
            target_keywords = context.get('target_keywords', [])
            
            # Perform comprehensive SEO analysis
            seo_analysis = self._analyze_seo(content, blog_title, outline, target_keywords)
            
            # Enhanced semantic keyword analysis
            semantic_keywords = await self._analyze_semantic_keywords(content, blog_title, target_keywords)
            
            # Content gap analysis
            content_gaps = await self._analyze_content_gaps(content, blog_title, target_keywords)
            
            # Generate AI-powered recommendations
            recommendations = await self._generate_ai_recommendations(seo_analysis, semantic_keywords, content_gaps)
            
            # Calculate enhanced SEO score
            seo_score = self._calculate_enhanced_seo_score(seo_analysis, semantic_keywords, content_gaps)
            
            result_data = {
                "seo_analysis": seo_analysis,
                "semantic_analysis": semantic_keywords,
                "content_gaps": content_gaps,
                "recommendations": recommendations,
                "seo_score": seo_score,
                "meta_title": self._generate_meta_title(blog_title, target_keywords),
                "meta_description": self._generate_meta_description(content, target_keywords),
                "suggested_keywords": self._extract_keywords(content, blog_title),
                "schema_markup": self._generate_schema_markup(content, blog_title),
                "technical_seo": self._analyze_technical_seo(content),
                "competitive_analysis": await self._analyze_competitive_landscape(target_keywords),
                "ranking_potential": self._assess_ranking_potential(seo_analysis, semantic_keywords)
            }
            
            logger.info(f"SEOAgent completed successfully. SEO Score: {seo_score}/100")
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except AgentExecutionError as e:
            logger.error(f"SEOAgent execution failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred in SEOAgent: {e}", exc_info=True)
            raise AgentExecutionError("SEOAgent", "execution", f"An unexpected error occurred: {e}")
    
    def _analyze_seo(self, content: str, blog_title: str, outline: List[str], target_keywords: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive SEO analysis.
        """
        analysis = {
            "content_length": len(content),
            "word_count": len(content.split()),
            "title_length": len(blog_title),
            "title_contains_keywords": False,
            "heading_structure": self._analyze_headings(content),
            "keyword_density": self._analyze_keyword_density(content, target_keywords),
            "readability_score": self._calculate_readability(content),
            "internal_linking": self._analyze_internal_links(content),
            "image_optimization": self._analyze_images(content),
            "meta_tags": self._analyze_meta_tags(content),
            "url_structure": self._analyze_url_structure(blog_title)
        }
        
        return analysis
    
    def _analyze_headings(self, content: str) -> Dict[str, Any]:
        """
        Analyze heading structure for SEO.
        """
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        h1_count = len(re.findall(r'^#\s+(.+)$', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+(.+)$', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+(.+)$', content, re.MULTILINE))
        
        return {
            "total_headings": len(headings),
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "has_proper_structure": h1_count == 1 and h2_count > 0,
            "headings": headings[:5]  # First 5 headings
        }
    
    def _analyze_keyword_density(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze keyword density and distribution.
        """
        content_lower = content.lower()
        word_count = len(content.split())
        
        keyword_analysis = {}
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            
            keyword_analysis[keyword] = {
                "count": count,
                "density": round(density, 2),
                "optimal_density": 0.5 <= density <= 2.5
            }
        
        return keyword_analysis
    
    def _calculate_readability(self, content: str) -> Dict[str, Any]:
        """
        Calculate readability score using Flesch Reading Ease.
        """
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        syllables = self._count_syllables(content)
        
        if len(sentences) == 0 or len(words) == 0:
            return {"score": 0, "level": "Unknown"}
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine reading level
        if flesch_score >= 90:
            level = "Very Easy"
        elif flesch_score >= 80:
            level = "Easy"
        elif flesch_score >= 70:
            level = "Fairly Easy"
        elif flesch_score >= 60:
            level = "Standard"
        elif flesch_score >= 50:
            level = "Fairly Difficult"
        elif flesch_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return {
            "score": round(flesch_score, 1),
            "level": level,
            "sentences": len(sentences),
            "words": len(words),
            "syllables": syllables
        }
    
    def _count_syllables(self, text: str) -> int:
        """
        Count syllables in text (simplified method).
        """
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(count, 1)
    
    def _analyze_internal_links(self, content: str) -> Dict[str, Any]:
        """
        Analyze internal linking structure.
        """
        # Regex for Markdown links: [text](url)
        markdown_links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', content)
        # Regex for basic HTML links: <a href="url">
        html_links = re.findall(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', content)
        
        all_links = markdown_links + html_links
        internal_links = [link for link in all_links if not re.match(r'^(http|https)', link)]
        external_links = [link for link in all_links if re.match(r'^(http|https)', link)]
        
        return {
            "total_links": len(all_links),
            "internal_links_count": len(internal_links),
            "external_links_count": len(external_links),
            "has_internal_linking": len(internal_links) > 0,
            "has_external_linking": len(external_links) > 0
        }
    
    def _analyze_images(self, content: str) -> Dict[str, Any]:
        """
        Analyze image optimization.
        """
        images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        
        return {
            "total_images": len(images),
            "images_with_alt": len([img for img in images if img[0].strip()]),
            "optimization_score": len([img for img in images if img[0].strip()]) / max(len(images), 1) * 100
        }
    
    def _analyze_meta_tags(self, content: str) -> Dict[str, Any]:
        """
        Analyze meta tag presence and quality.
        """
        # This would typically analyze HTML meta tags
        # For markdown content, we'll analyze content structure
        return {
            "has_meta_title": True,  # Assuming we generate one
            "has_meta_description": True,  # Assuming we generate one
            "has_structured_data": False,  # Would need schema markup
            "has_social_tags": False  # Would need Open Graph tags
        }
    
    def _analyze_url_structure(self, title: str) -> Dict[str, Any]:
        """
        Analyze URL structure based on title.
        """
        url_friendly = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
        url_friendly = re.sub(r'\s+', '-', url_friendly.strip())
        
        return {
            "suggested_url": url_friendly,
            "url_length": len(url_friendly),
            "is_optimal_length": 30 <= len(url_friendly) <= 60,
            "contains_keywords": True  # Assuming title contains keywords
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate SEO recommendations based on analysis.
        """
        recommendations = []
        
        # Content length recommendations
        if analysis["word_count"] < 300:
            recommendations.append("Increase content length to at least 300 words for better SEO")
        elif analysis["word_count"] > 2000:
            recommendations.append("Consider breaking long content into multiple articles")
        
        # Heading structure recommendations
        if not analysis["heading_structure"]["has_proper_structure"]:
            recommendations.append("Ensure proper heading hierarchy: one H1, multiple H2s")
        
        # Readability recommendations
        readability = analysis["readability_score"]
        if readability["score"] < 60:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        # Keyword density recommendations
        keyword_density = analysis["keyword_density"]
        for keyword, data in keyword_density.items():
            if data["density"] < 0.5:
                recommendations.append(f"Increase usage of keyword '{keyword}' naturally in content")
            elif data["density"] > 2.5:
                recommendations.append(f"Reduce overuse of keyword '{keyword}' to avoid keyword stuffing")
        
        # Internal linking recommendations
        if not analysis["internal_linking"]["has_internal_linking"]:
            recommendations.append("Add internal links to related content for better SEO")
        
        # Image optimization recommendations
        image_analysis = analysis["image_optimization"]
        if image_analysis["total_images"] > 0 and image_analysis["optimization_score"] < 100:
            recommendations.append("Add alt text to all images for better accessibility and SEO")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_seo_score(self, analysis: Dict[str, Any]) -> int:
        """
        Calculate overall SEO score (0-100).
        """
        score = 0
        max_score = 100
        
        # Content length (20 points)
        if 300 <= analysis["word_count"] <= 2000:
            score += 20
        elif analysis["word_count"] > 2000:
            score += 15
        elif analysis["word_count"] >= 200:
            score += 10
        
        # Heading structure (15 points)
        if analysis["heading_structure"]["has_proper_structure"]:
            score += 15
        
        # Readability (20 points)
        readability = analysis["readability_score"]
        if readability["score"] >= 70:
            score += 20
        elif readability["score"] >= 60:
            score += 15
        elif readability["score"] >= 50:
            score += 10
        
        # Keyword optimization (25 points)
        keyword_density = analysis["keyword_density"]
        keyword_score = 0
        for keyword, data in keyword_density.items():
            if data["optimal_density"]:
                keyword_score += 8
            elif data["density"] > 0:
                keyword_score += 4
        score += min(keyword_score, 25)
        
        # Internal linking (10 points)
        if analysis["internal_linking"]["has_internal_linking"]:
            score += 10
        
        # Image optimization (10 points)
        image_analysis = analysis["image_optimization"]
        if image_analysis["total_images"] == 0:
            score += 10  # No images to optimize
        elif image_analysis["optimization_score"] >= 80:
            score += 10
        elif image_analysis["optimization_score"] >= 60:
            score += 5
        
        return min(score, max_score)
    
    def _generate_meta_title(self, blog_title: str, target_keywords: List[str]) -> str:
        """
        Generate optimized meta title.
        """
        title = blog_title.strip()
        
        # Add primary keyword if not present
        if target_keywords and target_keywords[0].lower() not in title.lower():
            title = f"{title} - {target_keywords[0]}"
        
        # Ensure optimal length (50-60 characters)
        if len(title) > 60:
            title = title[:57] + "..."
        
        return title
    
    def _generate_meta_description(self, content: str, target_keywords: List[str]) -> str:
        """
        Generate optimized meta description.
        """
        # Extract first paragraph or create from content
        paragraphs = content.split('\n\n')
        description = ""
        
        for para in paragraphs:
            if para.strip() and not para.startswith('#'):
                description = para.strip()
                break
        
        if not description:
            description = content[:160]
        
        # Clean up description
        description = re.sub(r'[#*`]', '', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Ensure optimal length (150-160 characters)
        if len(description) > 160:
            description = description[:157] + "..."
        
        return description
    
    def _extract_keywords(self, content: str, blog_title: str) -> List[str]:
        """
        Extract potential keywords from content.
        """
        # Simple keyword extraction (in production, use more sophisticated methods)
        text = f"{blog_title} {content}".lower()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Use Counter for more efficient frequency counting
        word_counts = Counter(word for word in words if word not in stop_words)
        
        # Return the 10 most common keywords
        return [word for word, freq in word_counts.most_common(10)]
    
    def _sanitize_input(self, text: str, max_length: int = 4000) -> str:\n        \"\"\"Basic sanitization to prevent prompt injection.\"\"\"\n        text = text.strip()\n        # Basic removal of characters that might interfere with prompts\n        text = re.sub(r'[{}`<>]', '', text)\n        return text[:max_length]\n    \n    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for SEO analysis.
        """
        required_fields = ['content', 'blog_title']
        
        for field in required_fields:
            if not context.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    async def _analyze_semantic_keywords(self, content: str, title: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Enhanced semantic keyword analysis using AI.
        """
        if not self.openai_client:
            return self._fallback_keyword_analysis(content, title, target_keywords)
        
        try:
            prompt = f"""
            Analyze the following content for SEO keywords and semantic relevance:
            
            Title: {title}
            Content: {content[:2000]}...
            Target Keywords: {', '.join(target_keywords)}
            
            Provide a comprehensive keyword analysis including:
            1. Primary keyword opportunities
            2. Long-tail keyword variations
            3. Semantic keyword clusters
            4. Search intent classification
            5. Keyword difficulty assessment
            6. Content-keyword alignment score
            
            Return as JSON with these fields:
            - primary_keywords: list of main keywords
            - long_tail_keywords: list of long-tail variations
            - semantic_clusters: grouped related terms
            - search_intent: primary, secondary intents
            - keyword_gaps: missing relevant keywords
            - content_alignment_score: 0-100 score
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.warning(f"AI keyword analysis failed, using fallback: {e}")
            return self._fallback_keyword_analysis(content, title, target_keywords)
    
    def _fallback_keyword_analysis(self, content: str, title: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Fallback keyword analysis when AI is not available.
        """
        keywords = self._extract_keywords(content, title)
        return {
            "primary_keywords": keywords[:5],
            "long_tail_keywords": [kw for kw in keywords if len(kw.split()) > 2][:5],
            "semantic_clusters": {"main": keywords[:10]},
            "search_intent": "informational",
            "keyword_gaps": [],
            "content_alignment_score": 70
        }
    
    async def _analyze_content_gaps(self, content: str, title: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Identify content gaps and optimization opportunities.
        """
        if not self.openai_client:
            return {"gaps": [], "opportunities": [], "coverage_score": 75}
        
        try:
            prompt = f"""
            Analyze this content for SEO content gaps and opportunities:
            
            Title: {title}
            Content: {content[:2000]}...
            Target Keywords: {', '.join(target_keywords)}
            
            Identify:
            1. Missing topics that should be covered
            2. Thin content areas that need expansion
            3. Related subtopics to improve topical authority
            4. Content structure improvements
            5. E-E-A-T opportunities (Experience, Expertise, Authoritativeness, Trust)
            
            Return JSON with:
            - content_gaps: list of missing topics
            - expansion_opportunities: areas to expand
            - related_topics: relevant subtopics to add
            - structure_improvements: heading/organization suggestions
            - eat_opportunities: ways to improve E-E-A-T
            - coverage_score: 0-100 topical coverage score
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Content gap analysis failed: {e}")
            return {"gaps": [], "opportunities": [], "coverage_score": 75}
    
    async def _generate_ai_recommendations(self, seo_analysis: Dict[str, Any], semantic_keywords: Dict[str, Any], content_gaps: Dict[str, Any]) -> List[str]:
        """
        Generate AI-powered SEO recommendations.
        """
        if not self.openai_client:
            return self._generate_recommendations(seo_analysis)
        
        try:
            analysis_summary = {
                "word_count": seo_analysis.get("word_count", 0),
                "seo_score": self._calculate_seo_score(seo_analysis),
                "keyword_alignment": semantic_keywords.get("content_alignment_score", 70),
                "content_coverage": content_gaps.get("coverage_score", 75)
            }
            
            prompt = f"""
            Based on this SEO analysis, provide actionable optimization recommendations:
            
            Analysis Summary: {json.dumps(analysis_summary, indent=2)}
            Content Gaps: {json.dumps(content_gaps.get('content_gaps', [])[:5], indent=2)}
            Keyword Gaps: {json.dumps(semantic_keywords.get('keyword_gaps', [])[:5], indent=2)}
            
            Provide 5-8 specific, actionable SEO recommendations prioritized by impact.
            Focus on:
            1. Content optimization
            2. Keyword integration
            3. Technical improvements
            4. User experience enhancements
            5. E-E-A-T improvements
            
            Return as a JSON array of recommendation strings.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"AI recommendations failed, using fallback: {e}")
            return self._generate_recommendations(seo_analysis)
    
    def _generate_schema_markup(self, content: str, title: str) -> Dict[str, Any]:
        """
        Generate JSON-LD schema markup for the content.
        """
        # Extract article information
        word_count = len(content.split())
        
        # Estimate reading time (average 200 words per minute)
        reading_time_minutes = max(1, word_count // 200)
        
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": self._generate_meta_description(content, []),
            "wordCount": word_count,
            "datePublished": datetime.utcnow().isoformat(),
            "dateModified": datetime.utcnow().isoformat(),
            "author": {
                "@type": "Organization",
                "name": "CrediLinQ"
            },
            "publisher": {
                "@type": "Organization", 
                "name": "CrediLinQ"
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": f"https://example.com/blog/{self._generate_url_slug(title)}"
            },
            "timeRequired": f"PT{reading_time_minutes}M"
        }
        
        return schema
    
    def _generate_url_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title."""
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
        slug = re.sub(r'\s+', '-', slug.strip())
        return slug[:50]  # Limit length
    
    def _analyze_technical_seo(self, content: str) -> Dict[str, Any]:
        """
        Analyze technical SEO factors.
        """
        return {
            "content_structure": {
                "has_introduction": bool(re.search(r'^.{100,500}', content, re.DOTALL)),
                "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
                "paragraph_length": self._analyze_paragraph_length(content),
                "sentence_length": self._analyze_sentence_length(content)
            },
            "markup_opportunities": {
                "faq_sections": len(re.findall(r'\?', content)),
                "list_items": len(re.findall(r'^\s*[-*+]', content, re.MULTILINE)),
                "step_by_step": "step" in content.lower() and "process" in content.lower()
            },
            "accessibility": {
                "heading_hierarchy": self._check_heading_hierarchy(content),
                "list_structure": bool(re.search(r'^\s*[-*+]', content, re.MULTILINE))
            }
        }
    
    def _analyze_paragraph_length(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph length for readability."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        lengths = [len(p.split()) for p in paragraphs]
        
        if not lengths:
            return {"average": 0, "recommendation": "Add paragraph breaks"}
        
        avg_length = sum(lengths) / len(lengths)
        return {
            "average": round(avg_length, 1),
            "recommendation": "Good paragraph length" if 20 <= avg_length <= 150 else "Consider shorter paragraphs"
        }
    
    def _analyze_sentence_length(self, content: str) -> Dict[str, Any]:
        """Analyze sentence length for readability."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        lengths = [len(s.split()) for s in sentences]
        
        if not lengths:
            return {"average": 0, "recommendation": "Add more content"}
        
        avg_length = sum(lengths) / len(lengths)
        return {
            "average": round(avg_length, 1),
            "recommendation": "Good sentence length" if 15 <= avg_length <= 25 else "Consider shorter sentences"
        }
    
    def _check_heading_hierarchy(self, content: str) -> Dict[str, Any]:
        """Check if heading hierarchy is logical."""
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        hierarchy_issues = []
        prev_level = 0
        
        for heading_markup, heading_text in headings:
            level = len(heading_markup)
            if level > prev_level + 1:
                hierarchy_issues.append(f"Heading jump from H{prev_level} to H{level}")
            prev_level = level
        
        return {
            "is_logical": len(hierarchy_issues) == 0,
            "issues": hierarchy_issues
        }
    
    async def _analyze_competitive_landscape(self, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze competitive landscape for target keywords.
        """
        if not target_keywords:
            return {"competition_level": "unknown", "opportunities": []}
        
        # Simplified competitive analysis - in production, would use SEO APIs
        return {
            "competition_level": "medium",
            "keyword_difficulty": {kw: "medium" for kw in target_keywords[:3]},
            "opportunities": [
                "Target long-tail variations",
                "Focus on user intent optimization", 
                "Improve content depth and authority"
            ]
        }
    
    def _assess_ranking_potential(self, seo_analysis: Dict[str, Any], semantic_keywords: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the content's ranking potential.
        """
        base_score = self._calculate_seo_score(seo_analysis)
        keyword_score = semantic_keywords.get("content_alignment_score", 70)
        
        overall_score = (base_score + keyword_score) / 2
        
        if overall_score >= 80:
            potential = "High"
            timeframe = "1-3 months"
        elif overall_score >= 60:
            potential = "Medium" 
            timeframe = "3-6 months"
        else:
            potential = "Low"
            timeframe = "6+ months"
        
        return {
            "ranking_potential": potential,
            "estimated_timeframe": timeframe,
            "overall_score": round(overall_score, 1),
            "improvement_areas": self._identify_improvement_areas(seo_analysis, semantic_keywords)
        }
    
    def _identify_improvement_areas(self, seo_analysis: Dict[str, Any], semantic_keywords: Dict[str, Any]) -> List[str]:
        """Identify key areas for improvement."""
        areas = []
        
        if seo_analysis.get("word_count", 0) < 500:
            areas.append("Content length")
        
        if not seo_analysis.get("heading_structure", {}).get("has_proper_structure"):
            areas.append("Heading structure")
        
        if semantic_keywords.get("content_alignment_score", 100) < 70:
            areas.append("Keyword optimization")
        
        readability = seo_analysis.get("readability_score", {}).get("score", 100)
        if readability < 60:
            areas.append("Readability")
        
        return areas
    
    def _calculate_enhanced_seo_score(self, seo_analysis: Dict[str, Any], semantic_keywords: Dict[str, Any], content_gaps: Dict[str, Any]) -> int:
        """
        Calculate enhanced SEO score including semantic and content analysis.
        """
        base_score = self._calculate_seo_score(seo_analysis)
        
        # Semantic keyword score (0-20 points)
        keyword_alignment = semantic_keywords.get("content_alignment_score", 70)
        semantic_score = (keyword_alignment / 100) * 20
        
        # Content coverage score (0-15 points)
        coverage_score = content_gaps.get("coverage_score", 75)
        content_score = (coverage_score / 100) * 15
        
        # Technical SEO bonus (0-10 points)
        technical_bonus = 10  # Would be calculated based on technical analysis
        
        total_score = base_score + semantic_score + content_score
        return min(int(total_score), 100) 