"""
GEO Agent - Generates Generative Engine Optimization (GEO) metadata for content.
Optimizes content for AI Answer Engines like ChatGPT, Gemini, and Perplexity.
"""

from typing import Dict, Any, Optional, List
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..core.base_agent import BaseAgent, AgentResult, AgentExecutionContext, AgentMetadata, AgentType
from ...core.security import SecurityValidator


class GEOAgent(BaseAgent[Dict[str, Any]]):
    """
    Agent responsible for generating Generative Engine Optimization (GEO) metadata
    to maximize content findability, trustworthiness, and citability by AI Answer Engines.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        if metadata is None:
            metadata = AgentMetadata(
                agent_type=AgentType.CONTENT_OPTIMIZER,
                name="GEOAgent",
                description="Generates GEO metadata for AI Answer Engine optimization",
                capabilities=[
                    "geo_optimization",
                    "structured_data_generation",
                    "faq_creation",
                    "entity_mapping",
                    "citation_optimization",
                    "answer_engine_targeting"
                ],
                version="1.0.0"
            )
        
        super().__init__(metadata)
        self.security_validator = SecurityValidator()
        self.llm = None
    
    def _initialize(self):
        """Initialize the LLM and other resources."""
        try:
            from ...config.settings import get_settings
            settings = get_settings()
            
            self.llm = ChatOpenAI(
                model="gpt-4",  # Use GPT-4 for better structured output
                temperature=0.3,  # Lower temperature for more consistent structured output
                openai_api_key=settings.OPENAI_API_KEY
            )
            self.logger.info("GEOAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GEOAgent: {str(e)}")
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for GEO optimization."""
        super()._validate_input(input_data)
        
        required_fields = ["topic", "target_audience", "unique_angle", "author_brand"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Security validation
        for field in required_fields:
            if isinstance(input_data[field], str):
                self.security_validator.validate_input(input_data[field])
    
    def execute(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None,
        **kwargs
    ) -> AgentResult:
        """
        Generate GEO optimization package for content.
        
        Args:
            input_data: Dictionary containing:
                - topic: Main topic/title of content
                - target_audience: Target audience description
                - unique_angle: Unique perspective or key insight
                - author_brand: Author or brand name
                - company_context: Optional company context
            context: Execution context
            
        Returns:
            AgentResult: Result containing the GEO optimization package
        """
        try:
            # Initialize if not already done
            if self.llm is None:
                self._initialize()
            
            topic = input_data["topic"]
            target_audience = input_data["target_audience"]
            unique_angle = input_data["unique_angle"]
            author_brand = input_data["author_brand"]
            company_context = input_data.get("company_context", "")
            
            self.logger.info(f"Generating GEO package for topic: {topic}")
            
            # Generate GEO package
            geo_package = self._generate_geo_package(
                topic, target_audience, unique_angle, author_brand, company_context
            )
            
            result_data = {
                "geo_strategy": geo_package,
                "optimization_score": self._calculate_optimization_score(geo_package),
                "ai_engine_readiness": self._assess_ai_readiness(geo_package)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent_type": "geo_optimizer",
                    "topic": topic,
                    "optimization_level": "advanced"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate GEO package: {str(e)}")
            return AgentResult(
                success=False,
                error_message=str(e),
                error_code="GEO_GENERATION_FAILED"
            )

    def _generate_geo_package(
        self,
        topic: str,
        target_audience: str,
        unique_angle: str,
        author_brand: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Generate the complete GEO optimization package."""
        
        prompt = self._create_geo_prompt(topic, target_audience, unique_angle, author_brand, company_context)
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            geo_package = self._parse_geo_response(response.content.strip())
            
            self.logger.info("Successfully generated GEO package")
            return geo_package
            
        except Exception as e:
            self.logger.warning(f"LLM GEO generation failed: {str(e)}, using fallback")
            return self._create_fallback_geo_package(topic, target_audience, unique_angle, author_brand)
    
    def _create_geo_prompt(
        self, 
        topic: str, 
        target_audience: str, 
        unique_angle: str, 
        author_brand: str,
        company_context: str
    ) -> str:
        """Create the GEO optimization prompt."""
        
        context_section = f"\n**Company Context:** {company_context}" if company_context else ""
        
        return f"""Act as an expert in Generative Engine Optimization (GEO) and digital content strategy. Your mission is to transform a standard topic into a "GEO-Ready Content Package." This package must be structured for maximum findability, trustworthiness, and citability by AI Answer Engines (like Gemini, Perplexity, ChatGPT).

The output must be a single, complete JSON object. Do not include any explanatory text outside of the JSON.

**Primary Inputs:**
* **Topic:** "{topic}"
* **Target Audience:** "{target_audience}"
* **Unique Angle / Key Insight:** "{unique_angle}"
* **Author / Brand Name:** "{author_brand}"{context_section}

**Your Task:**
Using the inputs above, generate a comprehensive GEO package in the following JSON format.

**JSON Structure Request:**
{{
  "geo_strategy": {{
    "optimized_title": "A clear, question-answering title.",
    "executive_summary": "A 2-3 sentence, highly concise summary of the core answer. This is the TL;DR for the AI.",
    "key_takeaways": [
      "A bulleted list of 3-5 crucial, tweet-sized takeaways.",
      "Each takeaway should be a complete, self-contained thought."
    ],
    "core_article_outline": [
      "Section title framed as a question or a direct statement of value.",
      "Another section title designed to answer a follow-up question."
    ],
    "faq_for_structured_data": [
      {{
        "question": "The most likely conversational question a user would ask an AI on this topic.",
        "answer": "A direct, unambiguous, and concise answer. Perfect for FAQPage Schema."
      }},
      {{
        "question": "A follow-up or 'what about' question.",
        "answer": "Another direct and complete answer."
      }}
    ],
    "entity_and_concept_map": {{
      "concept_name_1": "A simple, one-sentence definition.",
      "concept_name_2": "Another simple definition."
    }},
    "citable_expert_quote": {{
      "quote": "A powerful, memorable quote that encapsulates the unique angle.",
      "attribution": "{author_brand}"
    }}
  }}
}}

**Instructions for Filling the JSON:**
1. **optimized_title:** Create a title that sounds like a direct answer to a user's problem.
2. **executive_summary:** This is the most critical part for quick AI parsing. Get straight to the point. State the problem and the unique solution immediately.
3. **key_takeaways:** Make these easily digestible facts. AI models love lists of facts.
4. **core_article_outline:** Structure the flow of the main content. Each section should be a logical building block in the argument. Frame them as answers.
5. **faq_for_structured_data:** Generate 3-5 Q&A pairs. These should address the most common user queries. The answers must be self-contained and definitive.
6. **entity_and_concept_map:** Identify the key terms or concepts in the topic. Define them simply. This helps the AI build its knowledge graph and see you as an authority who defines terms.
7. **citable_expert_quote:** Craft a single, powerful sentence that an AI can easily lift and feature as a blockquote to add expert opinion to its answer. It must be attributed.

Return ONLY the JSON object, nothing else:"""
    
    def _parse_geo_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the GEO package JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                geo_package = json.loads(json_str)
                
                # Validate the structure
                if "geo_strategy" in geo_package:
                    return geo_package["geo_strategy"]
                else:
                    return geo_package
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse GEO JSON: {str(e)}")
            raise
    
    def _create_fallback_geo_package(
        self, 
        topic: str, 
        target_audience: str, 
        unique_angle: str, 
        author_brand: str
    ) -> Dict[str, Any]:
        """Create a fallback GEO package when LLM fails."""
        
        self.logger.info("Using fallback GEO package generation")
        
        return {
            "optimized_title": f"How to {topic}: A Complete Guide for {target_audience}",
            "executive_summary": f"{topic} is crucial for {target_audience}. {unique_angle} This guide provides actionable insights to help you succeed.",
            "key_takeaways": [
                f"Understanding {topic} is essential for success",
                f"{unique_angle} sets this approach apart",
                f"Implementation requires a strategic approach",
                f"Results can be measured and optimized"
            ],
            "core_article_outline": [
                f"What is {topic}?",
                f"Why {topic} Matters for {target_audience}",
                f"How to Implement {topic} Successfully",
                f"Common Challenges and Solutions",
                f"Measuring Success and Next Steps"
            ],
            "faq_for_structured_data": [
                {
                    "question": f"What is {topic}?",
                    "answer": f"{topic} is a strategic approach that helps {target_audience} achieve better results through focused implementation."
                },
                {
                    "question": f"How does {topic} benefit {target_audience}?",
                    "answer": f"{topic} provides {target_audience} with practical tools and insights to improve their outcomes and efficiency."
                },
                {
                    "question": f"What makes this approach to {topic} unique?",
                    "answer": unique_angle
                }
            ],
            "entity_and_concept_map": {
                topic: f"The main subject matter focusing on practical implementation for {target_audience}",
                target_audience: f"The primary beneficiaries of {topic} strategies and insights"
            },
            "citable_expert_quote": {
                "quote": f"The key to successful {topic} implementation lies in understanding your specific context and adapting proven strategies accordingly.",
                "attribution": author_brand
            }
        }
    
    def _calculate_optimization_score(self, geo_package: Dict[str, Any]) -> int:
        """Calculate a GEO optimization score (0-100)."""
        score = 0
        
        # Check for required elements (40 points total)
        required_elements = [
            "optimized_title", "executive_summary", "key_takeaways",
            "core_article_outline", "faq_for_structured_data",
            "entity_and_concept_map", "citable_expert_quote"
        ]
        
        for element in required_elements:
            if element in geo_package and geo_package[element]:
                score += 6  # ~6 points per required element
        
        # Quality checks (60 points total)
        # Title optimization (10 points)
        title = geo_package.get("optimized_title", "")
        if any(word in title.lower() for word in ["how", "what", "why", "guide", "complete"]):
            score += 10
        
        # Executive summary quality (10 points)
        summary = geo_package.get("executive_summary", "")
        if len(summary.split()) >= 20 and len(summary.split()) <= 50:
            score += 10
        
        # Key takeaways completeness (10 points)
        takeaways = geo_package.get("key_takeaways", [])
        if isinstance(takeaways, list) and 3 <= len(takeaways) <= 5:
            score += 10
        
        # FAQ quality (10 points)
        faqs = geo_package.get("faq_for_structured_data", [])
        if isinstance(faqs, list) and len(faqs) >= 3:
            score += 10
        
        # Entity mapping (10 points)
        entities = geo_package.get("entity_and_concept_map", {})
        if isinstance(entities, dict) and len(entities) >= 2:
            score += 10
        
        # Expert quote presence (10 points)
        quote = geo_package.get("citable_expert_quote", {})
        if isinstance(quote, dict) and "quote" in quote and "attribution" in quote:
            score += 10
        
        return min(score, 100)
    
    def _assess_ai_readiness(self, geo_package: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for different AI answer engines."""
        
        readiness = {
            "chatgpt": 0,
            "gemini": 0,
            "perplexity": 0,
            "overall": 0
        }
        
        # ChatGPT optimization factors
        if geo_package.get("executive_summary"):
            readiness["chatgpt"] += 30
        if geo_package.get("key_takeaways"):
            readiness["chatgpt"] += 25
        if geo_package.get("faq_for_structured_data"):
            readiness["chatgpt"] += 25
        if geo_package.get("citable_expert_quote"):
            readiness["chatgpt"] += 20
        
        # Gemini optimization factors
        if geo_package.get("entity_and_concept_map"):
            readiness["gemini"] += 35
        if geo_package.get("core_article_outline"):
            readiness["gemini"] += 30
        if geo_package.get("optimized_title"):
            readiness["gemini"] += 35
        
        # Perplexity optimization factors
        if geo_package.get("citable_expert_quote"):
            readiness["perplexity"] += 40
        if geo_package.get("faq_for_structured_data"):
            readiness["perplexity"] += 35
        if geo_package.get("executive_summary"):
            readiness["perplexity"] += 25
        
        # Overall readiness
        readiness["overall"] = sum([readiness["chatgpt"], readiness["gemini"], readiness["perplexity"]]) // 3
        
        return readiness