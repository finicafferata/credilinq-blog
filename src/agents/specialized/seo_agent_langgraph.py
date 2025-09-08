"""
LangGraph-based SEO Agent with comprehensive SEO optimization workflow.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class SEOState(WorkflowState):
    """State for SEO LangGraph workflow."""
    content: str = ""
    target_keywords: List[str] = field(default_factory=list)
    target_audience: str = "general"
    content_type: str = "blog_post"
    domain_authority: float = 50.0
    
    # SEO Analysis
    keyword_analysis: Dict[str, Any] = field(default_factory=dict)
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    technical_seo: Dict[str, Any] = field(default_factory=dict)
    competitor_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization Results
    seo_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    optimized_content: str = ""
    meta_tags: Dict[str, str] = field(default_factory=dict)
    schema_markup: Dict[str, Any] = field(default_factory=dict)
    
    # SEO Scores
    keyword_density_score: float = 0.0
    readability_score: float = 0.0
    technical_seo_score: float = 0.0
    content_quality_score: float = 0.0
    overall_seo_score: float = 0.0

class SEOAgentLangGraph(LangGraphWorkflowBase[SEOState]):
    """LangGraph-based SEO Agent with comprehensive optimization workflow."""
    
    def __init__(self, workflow_name: str = "SEO_workflow"):
        super().__init__(workflow_name=workflow_name)
        self.metadata = AgentMetadata(
            agent_type=AgentType.SEO,
            name="SEOAgentLangGraph", 
            description="Advanced SEO optimizer with comprehensive analysis and optimization workflow"
        )
        logger.info("SEOAgentLangGraph initialized")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(SEOState)
        
        # Multi-phase SEO workflow
        workflow.add_node("keyword_analysis", self._keyword_analysis)
        workflow.add_node("content_analysis", self._content_analysis)
        workflow.add_node("technical_seo_analysis", self._technical_seo_analysis)
        workflow.add_node("competitor_analysis", self._competitor_analysis)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("optimize_content", self._optimize_content)
        workflow.add_node("generate_meta_tags", self._generate_meta_tags)
        workflow.add_node("create_schema_markup", self._create_schema_markup)
        workflow.add_node("final_seo_scoring", self._final_seo_scoring)
        
        # Workflow structure
        workflow.set_entry_point("keyword_analysis")
        workflow.add_edge("keyword_analysis", "content_analysis")
        workflow.add_edge("content_analysis", "technical_seo_analysis")
        workflow.add_edge("technical_seo_analysis", "competitor_analysis")
        workflow.add_edge("competitor_analysis", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "optimize_content")
        workflow.add_edge("optimize_content", "generate_meta_tags")
        workflow.add_edge("generate_meta_tags", "create_schema_markup")
        workflow.add_edge("create_schema_markup", "final_seo_scoring")
        workflow.set_finish_point("final_seo_scoring")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> SEOState:
        """Create initial workflow state."""
        return SEOState(
            content=input_data.get("content", ""),
            target_keywords=input_data.get("target_keywords", []),
            target_audience=input_data.get("target_audience", "general"),
            content_type=input_data.get("content_type", "blog_post"),
            domain_authority=input_data.get("domain_authority", 50.0),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="keyword_analysis"
        )
    
    def _keyword_analysis(self, state: SEOState) -> SEOState:
        """Analyze keyword density, placement, and opportunities."""
        logger.info("Analyzing keyword density and placement")
        
        content = state.content.lower()
        word_count = len(state.content.split())
        
        keyword_analysis = {}
        
        for keyword in state.target_keywords:
            keyword_lower = keyword.lower()
            occurrences = content.count(keyword_lower)
            density = (occurrences / word_count * 100) if word_count > 0 else 0
            
            # Analyze keyword placement
            in_title = keyword_lower in content[:100].lower()
            in_first_paragraph = keyword_lower in content[:300].lower()
            in_headings = any(keyword_lower in line for line in state.content.split('\n') if line.startswith('#'))
            
            keyword_analysis[keyword] = {
                "occurrences": occurrences,
                "density_percentage": round(density, 2),
                "optimal_density": self._calculate_optimal_density(keyword),
                "placement": {
                    "in_title": in_title,
                    "in_first_paragraph": in_first_paragraph,
                    "in_headings": in_headings
                },
                "status": self._evaluate_keyword_status(density, in_title, in_first_paragraph)
            }
        
        # Identify additional keyword opportunities
        long_tail_opportunities = self._identify_long_tail_opportunities(state.content, state.target_keywords)
        semantic_keywords = self._suggest_semantic_keywords(state.target_keywords)
        
        state.keyword_analysis = {
            "primary_keywords": keyword_analysis,
            "long_tail_opportunities": long_tail_opportunities,
            "semantic_keywords": semantic_keywords,
            "overall_keyword_score": self._calculate_keyword_score(keyword_analysis)
        }
        
        state.current_step = "content_analysis"
        return state
    
    def _content_analysis(self, state: SEOState) -> SEOState:
        """Analyze content structure and SEO factors."""
        logger.info("Analyzing content structure for SEO")
        
        content = state.content
        
        # Content structure analysis
        structure_analysis = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "paragraph_count": len(content.split('\n\n')),
            "heading_structure": self._analyze_heading_structure(content),
            "internal_links": content.count(']('),  # Markdown links
            "external_links": content.count('http'),
            "images": content.count('!['),  # Markdown images
            "lists": content.count('- ') + content.count('* ') + content.count('1.')
        }
        
        # Readability analysis
        readability = {
            "avg_sentence_length": self._calculate_avg_sentence_length(content),
            "reading_level": self._estimate_reading_level(content),
            "passive_voice_count": self._count_passive_voice(content),
            "transition_words": self._count_transition_words(content)
        }
        
        # Content quality indicators
        quality_indicators = {
            "unique_content_score": self._estimate_uniqueness(content),
            "depth_score": self._analyze_content_depth(content),
            "freshness_indicators": self._identify_freshness_signals(content),
            "expertise_signals": self._identify_expertise_signals(content)
        }
        
        state.content_analysis = {
            "structure": structure_analysis,
            "readability": readability,
            "quality": quality_indicators,
            "content_score": self._calculate_content_score(structure_analysis, readability, quality_indicators)
        }
        
        state.current_step = "technical_seo_analysis"
        return state
    
    def _technical_seo_analysis(self, state: SEOState) -> SEOState:
        """Analyze technical SEO factors."""
        logger.info("Analyzing technical SEO factors")
        
        content = state.content
        
        # Title tag analysis
        title_analysis = self._analyze_title_tag(content, state.target_keywords)
        
        # URL analysis (simulated)
        url_analysis = {
            "suggested_slug": self._generate_url_slug(content, state.target_keywords),
            "slug_length": "optimal",
            "contains_keywords": True,
            "readability": "high"
        }
        
        # Header structure analysis
        header_analysis = self._analyze_headers_for_seo(content, state.target_keywords)
        
        # Image optimization analysis
        image_analysis = {
            "images_found": content.count('!['),
            "alt_text_present": True,  # Simulated
            "optimized_filenames": False,
            "compression_needed": True
        }
        
        # Internal linking analysis
        linking_analysis = {
            "internal_links": content.count('](/'),
            "external_links": content.count('](http'),
            "anchor_text_optimization": "moderate",
            "link_distribution": "even"
        }
        
        state.technical_seo = {
            "title": title_analysis,
            "url": url_analysis,
            "headers": header_analysis,
            "images": image_analysis,
            "linking": linking_analysis,
            "technical_score": self._calculate_technical_score(title_analysis, header_analysis, image_analysis)
        }
        
        state.current_step = "competitor_analysis"
        return state
    
    def _competitor_analysis(self, state: SEOState) -> SEOState:
        """Analyze competitor content and SEO strategies."""
        logger.info("Analyzing competitor SEO strategies")
        
        # Simulated competitor analysis
        competitor_analysis = {
            "top_competitors": [
                {"domain": "competitor1.com", "authority": 65, "content_length": 1200},
                {"domain": "competitor2.com", "authority": 58, "content_length": 1500},
                {"domain": "competitor3.com", "authority": 72, "content_length": 900}
            ],
            "content_gaps": [
                "Detailed case studies missing",
                "Limited visual content",
                "No interactive elements"
            ],
            "keyword_gaps": [
                "Long-tail variations not covered",
                "Local SEO keywords missing",
                "Technical terminology underutilized"
            ],
            "content_opportunities": [
                "Create more comprehensive guides",
                "Add expert interviews",
                "Include more recent data and statistics"
            ]
        }
        
        state.competitor_analysis = competitor_analysis
        state.current_step = "generate_recommendations"
        return state
    
    def _generate_recommendations(self, state: SEOState) -> SEOState:
        """Generate comprehensive SEO recommendations."""
        logger.info("Generating SEO recommendations")
        
        recommendations = []
        
        # Keyword recommendations
        for keyword, data in state.keyword_analysis["primary_keywords"].items():
            if data["density_percentage"] < 1.0:
                recommendations.append({
                    "type": "keyword",
                    "priority": "high",
                    "issue": f"Low keyword density for '{keyword}' ({data['density_percentage']}%)",
                    "recommendation": f"Increase '{keyword}' density to 1-2% by naturally incorporating it in content",
                    "impact": "high"
                })
            
            if not data["placement"]["in_title"]:
                recommendations.append({
                    "type": "keyword",
                    "priority": "high",
                    "issue": f"Keyword '{keyword}' not in title",
                    "recommendation": f"Include '{keyword}' in the title/heading",
                    "impact": "high"
                })
        
        # Content recommendations
        content_data = state.content_analysis
        if content_data["structure"]["word_count"] < 600:
            recommendations.append({
                "type": "content",
                "priority": "medium",
                "issue": "Content length below optimal range",
                "recommendation": "Expand content to 800-1200 words for better SEO performance",
                "impact": "medium"
            })
        
        if not content_data["structure"]["heading_structure"]["has_h2"]:
            recommendations.append({
                "type": "structure", 
                "priority": "high",
                "issue": "Missing H2 headings",
                "recommendation": "Add H2 headings to improve content structure and readability",
                "impact": "high"
            })
        
        # Technical SEO recommendations
        tech_data = state.technical_seo
        if tech_data["images"]["images_found"] > 0 and not tech_data["images"]["alt_text_present"]:
            recommendations.append({
                "type": "technical",
                "priority": "high",
                "issue": "Images missing alt text",
                "recommendation": "Add descriptive alt text to all images including target keywords where appropriate",
                "impact": "high"
            })
        
        state.seo_recommendations = recommendations
        state.current_step = "optimize_content"
        return state
    
    def _optimize_content(self, state: SEOState) -> SEOState:
        """Apply SEO optimizations to content."""
        logger.info("Optimizing content based on recommendations")
        
        optimized_content = state.content
        
        # Apply keyword optimizations
        for keyword in state.target_keywords[:2]:  # Focus on top 2 keywords
            keyword_data = state.keyword_analysis["primary_keywords"].get(keyword, {})
            
            # Add keyword to title if missing
            if not keyword_data.get("placement", {}).get("in_title", False):
                if optimized_content.startswith('#'):
                    # Update existing title
                    lines = optimized_content.split('\n')
                    if keyword.lower() not in lines[0].lower():
                        lines[0] = f"{lines[0].rstrip()} - {keyword}"
                        optimized_content = '\n'.join(lines)
                else:
                    # Add title
                    optimized_content = f"# {keyword} Guide\n\n{optimized_content}"
            
            # Naturally incorporate keyword if density is low
            if keyword_data.get("density_percentage", 0) < 1.0:
                # Add keyword in first paragraph if not present
                paragraphs = optimized_content.split('\n\n')
                if len(paragraphs) > 1 and keyword.lower() not in paragraphs[1].lower():
                    # Insert keyword naturally in first paragraph
                    first_para = paragraphs[1]
                    if '. ' in first_para:
                        sentences = first_para.split('. ')
                        sentences[0] += f" related to {keyword}"
                        paragraphs[1] = '. '.join(sentences)
                        optimized_content = '\n\n'.join(paragraphs)
                        break
        
        # Add structure improvements
        if state.content_analysis["structure"]["word_count"] < 600:
            # Add conclusion section
            if "conclusion" not in optimized_content.lower():
                optimized_content += f"\n\n## Conclusion\n\nThis comprehensive guide covers the essential aspects of {state.target_keywords[0] if state.target_keywords else 'the topic'}. By following these insights, you can achieve better results and stay ahead of the competition."
        
        # Add H2 headings if missing
        if not state.content_analysis["structure"]["heading_structure"]["has_h2"]:
            paragraphs = optimized_content.split('\n\n')
            if len(paragraphs) >= 3:
                # Add H2 heading before the second major paragraph
                for i, para in enumerate(paragraphs[1:], 1):
                    if len(para.split()) > 50 and not para.startswith('#'):
                        paragraphs.insert(i, "## Key Insights")
                        break
                optimized_content = '\n\n'.join(paragraphs)
        
        state.optimized_content = optimized_content
        state.current_step = "generate_meta_tags"
        return state
    
    def _generate_meta_tags(self, state: SEOState) -> SEOState:
        """Generate optimized meta tags."""
        logger.info("Generating optimized meta tags")
        
        primary_keyword = state.target_keywords[0] if state.target_keywords else ""
        content_preview = state.optimized_content[:200].replace('\n', ' ')
        
        # Generate meta title
        title_parts = []
        if primary_keyword:
            title_parts.append(primary_keyword)
        if state.content_type == "blog_post":
            title_parts.append("Complete Guide")
        elif state.content_type == "product":
            title_parts.append("Best Solutions")
        
        meta_title = f"{' - '.join(title_parts)} | {state.target_audience.title()} Resource"
        if len(meta_title) > 60:
            meta_title = meta_title[:57] + "..."
        
        # Generate meta description
        meta_description = f"Discover comprehensive insights about {primary_keyword}. " if primary_keyword else ""
        meta_description += f"Expert guidance for {state.target_audience}. "
        
        # Add compelling call-to-action
        cta_options = ["Learn more today.", "Get started now.", "Read the full guide.", "Unlock the secrets."]
        meta_description += cta_options[0]
        
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        # Generate keywords meta tag (though less important now)
        keywords_list = state.target_keywords + state.keyword_analysis.get("semantic_keywords", [])[:5]
        
        state.meta_tags = {
            "title": meta_title,
            "description": meta_description,
            "keywords": ", ".join(keywords_list[:10]),
            "og_title": meta_title,
            "og_description": meta_description,
            "og_type": "article",
            "twitter_card": "summary_large_image",
            "twitter_title": meta_title,
            "twitter_description": meta_description
        }
        
        state.current_step = "create_schema_markup"
        return state
    
    def _create_schema_markup(self, state: SEOState) -> SEOState:
        """Create structured data markup."""
        logger.info("Creating schema markup")
        
        primary_keyword = state.target_keywords[0] if state.target_keywords else ""
        
        if state.content_type == "blog_post":
            schema_markup = {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": state.meta_tags["title"],
                "description": state.meta_tags["description"],
                "author": {
                    "@type": "Organization",
                    "name": "Expert Team"
                },
                "publisher": {
                    "@type": "Organization", 
                    "name": "Content Authority"
                },
                "datePublished": datetime.now().isoformat(),
                "dateModified": datetime.now().isoformat(),
                "mainEntityOfPage": {
                    "@type": "WebPage",
                    "@id": f"https://example.com/{primary_keyword.replace(' ', '-')}"
                },
                "keywords": state.meta_tags["keywords"]
            }
        elif state.content_type == "faq":
            schema_markup = {
                "@context": "https://schema.org",
                "@type": "FAQPage",
                "name": state.meta_tags["title"],
                "description": state.meta_tags["description"]
            }
        else:
            schema_markup = {
                "@context": "https://schema.org",
                "@type": "WebPage",
                "name": state.meta_tags["title"],
                "description": state.meta_tags["description"],
                "keywords": state.meta_tags["keywords"]
            }
        
        state.schema_markup = schema_markup
        state.current_step = "final_seo_scoring"
        return state
    
    def _final_seo_scoring(self, state: SEOState) -> SEOState:
        """Calculate final SEO scores."""
        logger.info("Calculating final SEO scores")
        
        # Calculate individual scores
        state.keyword_density_score = state.keyword_analysis["overall_keyword_score"]
        state.readability_score = min(state.content_analysis["readability"]["avg_sentence_length"] / 20, 1.0)
        state.technical_seo_score = state.technical_seo["technical_score"]
        state.content_quality_score = state.content_analysis["content_score"]
        
        # Calculate overall SEO score
        weights = {
            "keyword_density": 0.25,
            "readability": 0.20,
            "technical_seo": 0.30,
            "content_quality": 0.25
        }
        
        state.overall_seo_score = (
            state.keyword_density_score * weights["keyword_density"] +
            state.readability_score * weights["readability"] +
            state.technical_seo_score * weights["technical_seo"] +
            state.content_quality_score * weights["content_quality"]
        )
        
        # Update metadata
        state.metadata.update({
            "seo_optimization_complete": True,
            "overall_seo_score": state.overall_seo_score,
            "recommendations_count": len(state.seo_recommendations),
            "keywords_targeted": len(state.target_keywords),
            "meta_tags_generated": len(state.meta_tags),
            "schema_markup_created": bool(state.schema_markup),
            "content_optimized": bool(state.optimized_content)
        })
        
        state.current_step = "completed"
        return state
    
    # Helper methods
    def _calculate_optimal_density(self, keyword: str) -> float:
        """Calculate optimal keyword density based on keyword length."""
        word_count = len(keyword.split())
        if word_count == 1:
            return 1.5  # Single words can have higher density
        elif word_count == 2:
            return 1.0  # Two-word phrases
        else:
            return 0.8  # Long-tail keywords
    
    def _evaluate_keyword_status(self, density: float, in_title: bool, in_first_paragraph: bool) -> str:
        """Evaluate keyword optimization status."""
        if density < 0.5:
            return "under-optimized"
        elif density > 3.0:
            return "over-optimized" 
        elif in_title and in_first_paragraph and 1.0 <= density <= 2.0:
            return "well-optimized"
        elif in_title or in_first_paragraph:
            return "partially-optimized"
        else:
            return "needs-placement"
    
    def _identify_long_tail_opportunities(self, content: str, keywords: List[str]) -> List[str]:
        """Identify long-tail keyword opportunities."""
        # Simulated long-tail identification
        opportunities = []
        for keyword in keywords[:2]:  # Focus on main keywords
            opportunities.extend([
                f"best {keyword}",
                f"how to {keyword}",
                f"{keyword} guide",
                f"{keyword} tips",
                f"{keyword} examples"
            ])
        return opportunities[:5]
    
    def _suggest_semantic_keywords(self, keywords: List[str]) -> List[str]:
        """Suggest semantic keywords related to main keywords."""
        semantic_suggestions = []
        for keyword in keywords[:2]:
            # Simulated semantic keyword suggestions
            base_words = keyword.split()
            if base_words:
                semantic_suggestions.extend([
                    f"{base_words[0]} strategy",
                    f"{base_words[0]} implementation", 
                    f"{base_words[0]} best practices",
                    f"{base_words[0]} optimization"
                ])
        return semantic_suggestions[:8]
    
    def _calculate_keyword_score(self, keyword_analysis: Dict[str, Any]) -> float:
        """Calculate overall keyword optimization score."""
        if not keyword_analysis:
            return 0.0
        
        total_score = 0.0
        for keyword_data in keyword_analysis.values():
            density = keyword_data["density_percentage"]
            placement = keyword_data["placement"]
            
            # Score based on density (optimal range: 1-2%)
            if 1.0 <= density <= 2.0:
                density_score = 1.0
            elif 0.5 <= density < 1.0 or 2.0 < density <= 2.5:
                density_score = 0.7
            else:
                density_score = 0.3
            
            # Score based on placement
            placement_score = 0.0
            if placement["in_title"]:
                placement_score += 0.4
            if placement["in_first_paragraph"]:
                placement_score += 0.3
            if placement["in_headings"]:
                placement_score += 0.3
            
            total_score += (density_score * 0.6 + placement_score * 0.4)
        
        return total_score / len(keyword_analysis)
    
    def _analyze_heading_structure(self, content: str) -> Dict[str, Any]:
        """Analyze heading structure for SEO."""
        lines = content.split('\n')
        
        structure = {
            "has_h1": False,
            "has_h2": False,
            "has_h3": False,
            "h1_count": 0,
            "h2_count": 0,
            "h3_count": 0,
            "hierarchy_correct": True
        }
        
        for line in lines:
            if line.startswith('# '):
                structure["has_h1"] = True
                structure["h1_count"] += 1
            elif line.startswith('## '):
                structure["has_h2"] = True
                structure["h2_count"] += 1
            elif line.startswith('### '):
                structure["has_h3"] = True
                structure["h3_count"] += 1
        
        # Check if hierarchy is correct (H1 should be unique)
        if structure["h1_count"] > 1:
            structure["hierarchy_correct"] = False
            
        return structure
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length."""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _estimate_reading_level(self, content: str) -> str:
        """Estimate reading level (simplified)."""
        avg_sentence_length = self._calculate_avg_sentence_length(content)
        
        if avg_sentence_length < 15:
            return "Easy (Grade 6-8)"
        elif avg_sentence_length < 20:
            return "Moderate (Grade 9-12)"
        else:
            return "Difficult (College+)"
    
    def _count_passive_voice(self, content: str) -> int:
        """Count instances of passive voice (simplified)."""
        passive_indicators = [" was ", " were ", " been ", " being ", " is ", " are "]
        return sum(content.lower().count(indicator) for indicator in passive_indicators)
    
    def _count_transition_words(self, content: str) -> int:
        """Count transition words for readability."""
        transitions = [
            "however", "therefore", "furthermore", "moreover", "consequently",
            "additionally", "meanwhile", "nevertheless", "thus", "hence"
        ]
        return sum(content.lower().count(transition) for transition in transitions)
    
    def _estimate_uniqueness(self, content: str) -> float:
        """Estimate content uniqueness (simplified)."""
        # Simulated uniqueness score based on content characteristics
        unique_phrases = len(set(content.split('. ')))
        total_phrases = len(content.split('. '))
        
        if total_phrases == 0:
            return 0.0
        
        return min(unique_phrases / total_phrases, 1.0)
    
    def _analyze_content_depth(self, content: str) -> float:
        """Analyze content depth and comprehensiveness."""
        depth_indicators = [
            "example", "case study", "research", "study", "data", "statistics",
            "expert", "analysis", "detailed", "comprehensive", "thorough"
        ]
        
        content_lower = content.lower()
        found_indicators = sum(1 for indicator in depth_indicators if indicator in content_lower)
        
        return min(found_indicators / 8, 1.0)  # Normalize to 0-1 scale
    
    def _identify_freshness_signals(self, content: str) -> List[str]:
        """Identify content freshness signals."""
        freshness_signals = []
        content_lower = content.lower()
        
        current_year = datetime.now().year
        if str(current_year) in content:
            freshness_signals.append("Current year mentioned")
        
        fresh_indicators = ["latest", "recent", "new", "updated", "current", "today"]
        for indicator in fresh_indicators:
            if indicator in content_lower:
                freshness_signals.append(f"Uses '{indicator}' indicating freshness")
                break
        
        return freshness_signals
    
    def _identify_expertise_signals(self, content: str) -> List[str]:
        """Identify expertise and authority signals."""
        expertise_signals = []
        content_lower = content.lower()
        
        authority_indicators = [
            "research shows", "studies indicate", "expert opinion", "professional experience",
            "years of experience", "certified", "qualified", "proven", "tested"
        ]
        
        for indicator in authority_indicators:
            if indicator in content_lower:
                expertise_signals.append(f"Authority signal: '{indicator}'")
        
        return expertise_signals[:3]  # Limit to top 3
    
    def _calculate_content_score(self, structure: Dict, readability: Dict, quality: Dict) -> float:
        """Calculate overall content quality score."""
        structure_score = 0.0
        
        # Structure scoring
        if 800 <= structure["word_count"] <= 1500:
            structure_score += 0.3
        elif 600 <= structure["word_count"] <= 2000:
            structure_score += 0.2
        else:
            structure_score += 0.1
            
        if structure["heading_structure"]["has_h2"]:
            structure_score += 0.2
        if structure["heading_structure"]["hierarchy_correct"]:
            structure_score += 0.1
        if structure["internal_links"] > 0:
            structure_score += 0.1
        if structure["images"] > 0:
            structure_score += 0.1
        
        # Readability scoring
        readability_score = 0.0
        if 15 <= readability["avg_sentence_length"] <= 20:
            readability_score += 0.3
        elif 10 <= readability["avg_sentence_length"] <= 25:
            readability_score += 0.2
        else:
            readability_score += 0.1
            
        if readability["transition_words"] > 3:
            readability_score += 0.2
        
        # Quality scoring
        quality_score = quality["unique_content_score"] * 0.3 + quality["depth_score"] * 0.2
        
        return min(structure_score + readability_score + quality_score, 1.0)
    
    def _analyze_title_tag(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze title tag optimization."""
        lines = content.split('\n')
        title_line = next((line for line in lines if line.startswith('# ')), "")
        
        if not title_line:
            return {
                "present": False,
                "length": 0,
                "contains_keywords": False,
                "optimization_score": 0.0
            }
        
        title = title_line[2:]  # Remove '# '
        title_length = len(title)
        
        # Check if title contains keywords
        contains_keywords = any(keyword.lower() in title.lower() for keyword in keywords)
        
        # Calculate optimization score
        score = 0.0
        if 30 <= title_length <= 60:
            score += 0.4
        elif 20 <= title_length <= 70:
            score += 0.2
        
        if contains_keywords:
            score += 0.4
        
        if title_length > 0:
            score += 0.2
        
        return {
            "present": True,
            "title": title,
            "length": title_length,
            "contains_keywords": contains_keywords,
            "optimization_score": score
        }
    
    def _generate_url_slug(self, content: str, keywords: List[str]) -> str:
        """Generate SEO-friendly URL slug."""
        if keywords:
            primary_keyword = keywords[0]
            slug = primary_keyword.lower().replace(' ', '-')
            # Add content type if applicable
            if "guide" in content.lower()[:200]:
                slug += "-guide"
            elif "tips" in content.lower()[:200]:
                slug += "-tips"
            
            return slug
        
        # Fallback: use first heading
        lines = content.split('\n')
        title_line = next((line for line in lines if line.startswith('# ')), "")
        if title_line:
            return title_line[2:].lower().replace(' ', '-')[:50]
        
        return "optimized-content"
    
    def _analyze_headers_for_seo(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze headers for SEO optimization."""
        lines = content.split('\n')
        headers = {
            "h1": [line[2:] for line in lines if line.startswith('# ')],
            "h2": [line[3:] for line in lines if line.startswith('## ')],
            "h3": [line[4:] for line in lines if line.startswith('### ')]
        }
        
        # Count keyword usage in headers
        keyword_usage = 0
        all_headers = headers["h1"] + headers["h2"] + headers["h3"]
        
        for header in all_headers:
            for keyword in keywords:
                if keyword.lower() in header.lower():
                    keyword_usage += 1
                    break
        
        return {
            "header_count": len(all_headers),
            "h1_count": len(headers["h1"]),
            "h2_count": len(headers["h2"]),
            "h3_count": len(headers["h3"]),
            "keyword_usage_count": keyword_usage,
            "keyword_usage_percentage": (keyword_usage / len(all_headers) * 100) if all_headers else 0
        }
    
    def _calculate_technical_score(self, title_analysis: Dict, header_analysis: Dict, image_analysis: Dict) -> float:
        """Calculate technical SEO score."""
        score = 0.0
        
        # Title optimization
        score += title_analysis["optimization_score"] * 0.4
        
        # Header optimization
        if header_analysis["h2_count"] > 0:
            score += 0.2
        if header_analysis["keyword_usage_percentage"] > 30:
            score += 0.2
        
        # Image optimization
        if image_analysis["images_found"] > 0:
            if image_analysis["alt_text_present"]:
                score += 0.1
            else:
                score += 0.05  # Partial credit
        else:
            score += 0.1  # No images is okay
        
        # Basic structure
        score += 0.1  # Base score for having structure
        
        return min(score, 1.0)