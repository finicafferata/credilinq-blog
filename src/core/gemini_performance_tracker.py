"""
Gemini API Performance Tracker - Capture cost, tokens, and usage metrics for Gemini API calls.
Integrates with LangGraph execution for real-time performance tracking.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

import google.generativeai as genai
from google.ai.generativelanguage_v1beta import GenerationConfig

from src.config.database import db_config

logger = logging.getLogger(__name__)

@dataclass
class GeminiUsageMetrics:
    """Gemini API usage metrics extracted from response."""
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int
    cached_content_token_count: int = 0
    model_name: str = ""
    generation_config: Optional[Dict[str, Any]] = None
    finish_reason: str = "STOP"
    safety_ratings: List[Dict[str, Any]] = None

@dataclass
class GeminiCostCalculation:
    """Gemini cost calculation based on current pricing."""
    input_cost: float
    output_cost: float
    total_cost: float
    pricing_tier: str  # "flash" or "pro"
    input_tokens: int
    output_tokens: int

class GeminiPerformanceTracker:
    """
    Real-time performance tracking for Gemini API calls within LangGraph workflows.
    Captures tokens, cost, model usage, and decision-making metrics.
    """
    
    # Current Gemini pricing (as of 2024) - Update as needed
    PRICING = {
        "gemini-1.5-flash": {
            "input_cost_per_1k": 0.000075,  # $0.075 per 1M tokens
            "output_cost_per_1k": 0.0003,   # $0.30 per 1M tokens
            "tier": "flash"
        },
        "gemini-1.5-pro": {
            "input_cost_per_1k": 0.00125,   # $1.25 per 1M tokens
            "output_cost_per_1k": 0.005,    # $5.00 per 1M tokens
            "tier": "pro"
        },
        "gemini-1.0-pro": {
            "input_cost_per_1k": 0.0005,    # $0.50 per 1M tokens
            "output_cost_per_1k": 0.0015,   # $1.50 per 1M tokens
            "tier": "pro"
        }
    }
    
    def __init__(self):
        self.active_trackings: Dict[str, Dict[str, Any]] = {}
        self.db_connection = None
    
    async def track_gemini_execution_start(
        self,
        agent_name: str,
        agent_type: str,
        model_name: str,
        prompt_text: str,
        campaign_id: Optional[str] = None,
        blog_post_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a Gemini API execution.
        
        Returns:
            execution_id: Unique identifier for this tracking session
        """
        execution_id = f"gemini_{int(time.time() * 1000)}_{hash(agent_name) % 10000}"
        
        tracking_data = {
            "execution_id": execution_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "model_name": model_name,
            "start_time": datetime.utcnow(),
            "prompt_text": prompt_text[:1000],  # Store first 1000 chars for analysis
            "prompt_length": len(prompt_text),
            "campaign_id": campaign_id,
            "blog_post_id": blog_post_id,
            "workflow_id": workflow_id,
            "metadata": metadata or {},
            "status": "running"
        }
        
        self.active_trackings[execution_id] = tracking_data
        
        # Start async database insert (non-blocking)
        asyncio.create_task(self._insert_agent_performance_start(tracking_data))
        
        logger.debug(f"Started Gemini tracking for {agent_name}: {execution_id}")
        return execution_id
    
    async def track_gemini_execution_end(
        self,
        execution_id: str,
        response: Any,  # Gemini GenerateContentResponse
        status: str = "success",
        error_message: Optional[str] = None,
        error_code: Optional[str] = None
    ) -> Optional[GeminiUsageMetrics]:
        """
        Complete tracking of a Gemini API execution with response analysis.
        
        Args:
            execution_id: The tracking session identifier
            response: Gemini API response object
            status: "success", "failed", or "cancelled"
            error_message: Error message if failed
            error_code: Error code if failed
            
        Returns:
            GeminiUsageMetrics if successful, None if tracking data not found
        """
        if execution_id not in self.active_trackings:
            logger.warning(f"Tracking data not found for execution_id: {execution_id}")
            return None
        
        tracking_data = self.active_trackings[execution_id]
        end_time = datetime.utcnow()
        duration_ms = (end_time - tracking_data["start_time"]).total_seconds() * 1000
        
        usage_metrics = None
        cost_calculation = None
        
        if status == "success" and response:
            # Extract usage metrics from Gemini response
            usage_metrics = self._extract_usage_metrics(response, tracking_data["model_name"])
            
            if usage_metrics:
                # Calculate costs based on token usage
                cost_calculation = self._calculate_gemini_cost(
                    model_name=tracking_data["model_name"],
                    input_tokens=usage_metrics.prompt_token_count,
                    output_tokens=usage_metrics.candidates_token_count
                )
        
        # Update tracking data
        tracking_data.update({
            "end_time": end_time,
            "duration_ms": duration_ms,
            "status": status,
            "error_message": error_message,
            "error_code": error_code,
            "usage_metrics": usage_metrics.__dict__ if usage_metrics else None,
            "cost_calculation": cost_calculation.__dict__ if cost_calculation else None
        })
        
        # Async database update (non-blocking)
        asyncio.create_task(self._update_agent_performance_end(tracking_data))
        
        # Clean up active tracking
        del self.active_trackings[execution_id]
        
        logger.debug(f"Completed Gemini tracking: {execution_id}, tokens: {usage_metrics.total_token_count if usage_metrics else 0}")
        return usage_metrics
    
    def _extract_usage_metrics(self, response: Any, model_name: str) -> Optional[GeminiUsageMetrics]:
        """Extract usage metrics from Gemini response."""
        try:
            # Access usage metadata from Gemini response
            usage_metadata = getattr(response, 'usage_metadata', None)
            if not usage_metadata:
                logger.warning("No usage_metadata found in Gemini response")
                return None
            
            # Extract token counts
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', prompt_tokens + candidates_tokens)
            cached_tokens = getattr(usage_metadata, 'cached_content_token_count', 0)
            
            # Extract candidate information
            finish_reason = "STOP"
            safety_ratings = []
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                if hasattr(candidate, 'safety_ratings'):
                    safety_ratings = [
                        {
                            "category": str(rating.category),
                            "probability": str(rating.probability)
                        }
                        for rating in candidate.safety_ratings
                    ]
            
            return GeminiUsageMetrics(
                prompt_token_count=prompt_tokens,
                candidates_token_count=candidates_tokens,
                total_token_count=total_tokens,
                cached_content_token_count=cached_tokens,
                model_name=model_name,
                finish_reason=finish_reason,
                safety_ratings=safety_ratings
            )
            
        except Exception as e:
            logger.error(f"Error extracting Gemini usage metrics: {e}")
            return None
    
    def _calculate_gemini_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> GeminiCostCalculation:
        """Calculate cost based on Gemini pricing."""
        # Normalize model name for pricing lookup
        normalized_model = model_name.lower()
        for pricing_model in self.PRICING:
            if pricing_model in normalized_model:
                pricing = self.PRICING[pricing_model]
                break
        else:
            # Default to flash pricing if model not found
            pricing = self.PRICING["gemini-1.5-flash"]
            logger.warning(f"Unknown Gemini model {model_name}, using flash pricing")
        
        # Calculate costs (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000) * pricing["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_cost_per_1k"]
        total_cost = input_cost + output_cost
        
        return GeminiCostCalculation(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            pricing_tier=pricing["tier"],
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    async def _insert_agent_performance_start(self, tracking_data: Dict[str, Any]) -> None:
        """Insert initial performance record in database."""
        try:
            with db_config.get_db_connection() as conn:
                await conn.execute("""
                    INSERT INTO agent_performance (
                        execution_id, agent_name, agent_type, campaign_id, blog_post_id,
                        start_time, status, model_used, prompt_length, workflow_id, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, (
                    tracking_data["execution_id"],
                    tracking_data["agent_name"],
                    tracking_data["agent_type"],
                    tracking_data.get("campaign_id"),
                    tracking_data.get("blog_post_id"),
                    tracking_data["start_time"],
                    tracking_data["status"],
                    tracking_data["model_name"],
                    tracking_data["prompt_length"],
                    tracking_data.get("workflow_id"),
                    json.dumps(tracking_data["metadata"])
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error inserting agent performance start: {e}")
    
    async def _update_agent_performance_end(self, tracking_data: Dict[str, Any]) -> None:
        """Update performance record with completion data."""
        try:
            with db_config.get_db_connection() as conn:
                cost = 0
                input_tokens = 0
                output_tokens = 0
                
                if tracking_data.get("cost_calculation"):
                    cost_calc = tracking_data["cost_calculation"]
                    cost = cost_calc["total_cost"]
                    input_tokens = cost_calc["input_tokens"]
                    output_tokens = cost_calc["output_tokens"]
                
                await conn.execute("""
                    UPDATE agent_performance SET 
                        end_time = $2,
                        duration = $3,
                        status = $4,
                        input_tokens = $5,
                        output_tokens = $6,
                        cost = $7,
                        error_message = $8,
                        error_code = $9,
                        response_metadata = $10
                    WHERE execution_id = $1
                """, (
                    tracking_data["execution_id"],
                    tracking_data["end_time"],
                    int(tracking_data["duration_ms"]),
                    tracking_data["status"],
                    input_tokens,
                    output_tokens,
                    cost,
                    tracking_data.get("error_message"),
                    tracking_data.get("error_code"),
                    json.dumps({
                        "usage_metrics": tracking_data.get("usage_metrics"),
                        "cost_calculation": tracking_data.get("cost_calculation")
                    })
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating agent performance end: {e}")
    
    @contextmanager
    def track_gemini_call(
        self,
        agent_name: str,
        agent_type: str,
        model_name: str,
        prompt_text: str,
        **kwargs
    ):
        """
        Context manager for tracking Gemini API calls.
        
        Usage:
            with tracker.track_gemini_call("writer", "content", "gemini-1.5-pro", prompt) as tracking:
                response = model.generate_content(prompt)
                tracking.set_response(response)
        """
        class TrackingContext:
            def __init__(self, tracker, execution_id):
                self.tracker = tracker
                self.execution_id = execution_id
                self.response = None
                self.error = None
            
            def set_response(self, response):
                self.response = response
            
            def set_error(self, error):
                self.error = error
        
        # Start tracking
        loop = asyncio.get_event_loop()
        execution_id = loop.run_until_complete(
            self.track_gemini_execution_start(
                agent_name, agent_type, model_name, prompt_text, **kwargs
            )
        )
        
        tracking_context = TrackingContext(self, execution_id)
        
        try:
            yield tracking_context
            
            # Complete tracking
            if tracking_context.error:
                loop.run_until_complete(
                    self.track_gemini_execution_end(
                        execution_id, None, "failed", str(tracking_context.error)
                    )
                )
            else:
                loop.run_until_complete(
                    self.track_gemini_execution_end(
                        execution_id, tracking_context.response, "success"
                    )
                )
        except Exception as e:
            # Handle unexpected errors
            loop.run_until_complete(
                self.track_gemini_execution_end(
                    execution_id, None, "failed", str(e), "UNEXPECTED_ERROR"
                )
            )
            raise

# Global instance
global_gemini_tracker = GeminiPerformanceTracker()


def estimate_gemini_tokens(text: str) -> int:
    """
    Estimate token count for Gemini (rough approximation).
    Gemini tokenization is similar to OpenAI but this provides a reasonable estimate.
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return max(1, len(text) // 4)


def calculate_gemini_cost_estimate(input_text: str, output_text: str, model_name: str = "gemini-1.5-flash") -> float:
    """Quick cost estimate for Gemini API usage."""
    input_tokens = estimate_gemini_tokens(input_text)
    output_tokens = estimate_gemini_tokens(output_text)
    
    tracker = GeminiPerformanceTracker()
    cost_calc = tracker._calculate_gemini_cost(model_name, input_tokens, output_tokens)
    return cost_calc.total_cost