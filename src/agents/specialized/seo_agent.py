"""
SEO Agent - Analyzes and optimizes content for search engines
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, AgentType, AgentResult, AgentMetadata
from ...core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class SEOAgent(BaseAgent):
    """
    Agent specialized in SEO analysis and optimization.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None, name: str = "SEOAgent", description: str = "Analyzes and optimizes content for SEO"):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.SEO,
                name=name,
                description=description
            )
        super().__init__(metadata=metadata)
        self.agent_type = AgentType.SEO
        
    def execute(self, context: Dict[str, Any]) -> AgentResult:
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
            logger.info(f"SEOAgent executing for blog: {context.get('blog_title', 'Unknown')}")
            
            content = context.get('content', '')
            blog_title = context.get('blog_title', '')
            outline = context.get('outline', [])
            target_keywords = context.get('target_keywords', [])
            
            if not content and not blog_title:
                raise AgentExecutionError("No content or title provided for SEO analysis")
            
            # Perform SEO analysis
            seo_analysis = self._analyze_seo(content, blog_title, outline, target_keywords)
            
            # Generate SEO recommendations
            recommendations = self._generate_recommendations(seo_analysis)
            
            # Calculate SEO score
            seo_score = self._calculate_seo_score(seo_analysis)
            
            result_data = {
                "seo_analysis": seo_analysis,
                "recommendations": recommendations,
                "seo_score": seo_score,
                "meta_title": self._generate_meta_title(blog_title, target_keywords),
                "meta_description": self._generate_meta_description(content, target_keywords),
                "suggested_keywords": self._extract_keywords(content, blog_title)
            }
            
            logger.info(f"SEOAgent completed successfully. SEO Score: {seo_score}/100")
            
            return AgentResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"SEOAgent execution failed: {str(e)}")
            raise AgentExecutionError("SEOAgent", "execution", str(e))
    
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
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        internal_links = [link for link in links if not link[1].startswith('http')]
        
        return {
            "total_links": len(links),
            "internal_links": len(internal_links),
            "external_links": len(links) - len(internal_links),
            "has_internal_linking": len(internal_links) > 0
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
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context for SEO analysis.
        """
        required_fields = ['content', 'blog_title']
        
        for field in required_fields:
            if not context.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True 