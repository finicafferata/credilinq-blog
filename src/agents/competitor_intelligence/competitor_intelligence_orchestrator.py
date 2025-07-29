"""
Competitor Intelligence Orchestrator - Main coordinator for the multi-agent system.
Orchestrates all 6 specialized agents to provide comprehensive competitive intelligence.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import json

from ..core.base_agent import BaseAgent
from .models import (
    Competitor, ContentItem, Trend, ContentGap, CompetitorInsight,
    MarketAnalysis, Alert, Industry, MonitoringConfig, GapAnalysisRequest,
    CompetitorIntelligenceReport, AlertSubscription
)
from .content_monitoring_agent import ContentMonitoringAgent
from .trend_analysis_agent import TrendAnalysisAgent
from .gap_identification_agent import GapIdentificationAgent
from .performance_analysis_agent import PerformanceAnalysisAgent
from .strategic_insights_agent import StrategicInsightsAgent
from .alert_orchestration_agent import AlertOrchestrationAgent
from ...core.monitoring import metrics, async_performance_tracker
from ...core.cache import cache

class CompetitorIntelligenceOrchestrator(BaseAgent):
    """
    Main orchestrator for the competitive intelligence multi-agent system.
    Coordinates 6 specialized agents to provide comprehensive competitive intelligence.
    """
    
    def __init__(self):
        super().__init__(
            agent_type="competitor_intelligence_orchestrator",
            capabilities=[
                "multi_agent_coordination",
                "competitive_intelligence_synthesis",
                "workflow_orchestration",
                "comprehensive_reporting",
                "real_time_monitoring",
                "strategic_analysis"
            ]
        )
        
        # Initialize all specialized agents
        self.content_monitor = ContentMonitoringAgent()
        self.trend_analyzer = TrendAnalysisAgent()
        self.gap_identifier = GapIdentificationAgent()
        self.performance_analyzer = PerformanceAnalysisAgent()
        self.strategic_insights = StrategicInsightsAgent()
        self.alert_orchestrator = AlertOrchestrationAgent()
        
        # Orchestration configuration
        self.orchestration_config = {
            "max_concurrent_competitors": 10,
            "analysis_batch_size": 5,
            "cache_duration_hours": 2,
            "full_analysis_interval_hours": 24,
            "incremental_analysis_interval_hours": 4,
            "alert_check_interval_minutes": 15
        }
        
        # Workflow state
        self.workflow_state = {
            "last_full_analysis": None,
            "last_incremental_analysis": None,
            "active_monitoring_sessions": {},
            "cached_results": {}
        }
    
    async def run_comprehensive_analysis(
        self,
        competitors: List[Competitor],
        industry: Industry,
        your_content_topics: List[str] = None,
        analysis_depth: str = "standard"
    ) -> CompetitorIntelligenceReport:
        """
        Run comprehensive competitive intelligence analysis using all agents.
        This is the main entry point for full competitive analysis.
        """
        
        async with async_performance_tracker("comprehensive_ci_analysis"):
            self.logger.info(f"Starting comprehensive analysis for {len(competitors)} competitors in {industry.value}")
            
            # Step 1: Content Monitoring - Discover and analyze competitor content
            self.logger.info("Step 1: Starting content monitoring...")
            competitor_content = await self._orchestrate_content_monitoring(
                competitors, 
                analysis_depth
            )
            
            # Step 2: Trend Analysis - Identify trending topics and patterns
            self.logger.info("Step 2: Starting trend analysis...")
            all_content = []
            for content_list in competitor_content.values():
                all_content.extend(content_list)
            
            market_trends = await self.trend_analyzer.analyze_trending_topics(
                all_content,
                industry,
                time_window_days=30
            )
            
            # Step 3: Gap Identification - Find content opportunities
            self.logger.info("Step 3: Starting gap identification...")
            gap_request = GapAnalysisRequest(
                competitors=[comp.id for comp in competitors],
                your_content_topics=your_content_topics or [],
                industry=industry,
                analysis_depth=analysis_depth
            )
            
            content_gaps = await self.gap_identifier.identify_content_gaps(
                gap_request,
                competitor_content,
                market_trends
            )
            
            # Step 4: Performance Analysis - Benchmark competitor performance
            self.logger.info("Step 4: Starting performance analysis...")
            performance_analysis = await self.performance_analyzer.analyze_competitor_performance(
                competitors,
                competitor_content,
                analysis_period_days=90
            )
            
            # Step 5: Strategic Insights - Generate high-level strategic intelligence
            self.logger.info("Step 5: Generating strategic insights...")
            strategic_insights = await self.strategic_insights.generate_strategic_insights(
                competitors,
                market_trends,
                content_gaps,
                performance_analysis,
                industry
            )
            
            # Step 6: Generate Market Analysis
            self.logger.info("Step 6: Generating market analysis...")
            market_analysis = await self.trend_analyzer.generate_market_analysis(
                market_trends,
                all_content,
                industry
            )
            
            # Step 7: Generate Comprehensive Report
            self.logger.info("Step 7: Compiling comprehensive report...")
            intelligence_report = await self.strategic_insights.generate_competitive_intelligence_report(
                strategic_insights,
                market_analysis,
                competitors,
                market_trends,
                content_gaps
            )
            
            # Update workflow state
            self.workflow_state["last_full_analysis"] = datetime.utcnow()
            self._cache_analysis_results({
                "competitors": competitors,
                "content": competitor_content,
                "trends": market_trends,
                "gaps": content_gaps,
                "performance": performance_analysis,
                "insights": strategic_insights,
                "report": intelligence_report
            })
            
            # Track comprehensive metrics
            metrics.increment_counter(
                "comprehensive_analysis.completed",
                tags={
                    "industry": industry.value,
                    "competitors": str(len(competitors)),
                    "content_items": str(len(all_content)),
                    "trends_found": str(len(market_trends)),
                    "gaps_identified": str(len(content_gaps)),
                    "insights_generated": str(len(strategic_insights))
                }
            )
            
            self.logger.info(f"Comprehensive analysis completed. Report ID: {intelligence_report.report_id}")
            
            return intelligence_report
    
    async def run_incremental_monitoring(
        self,
        competitors: List[Competitor],
        alert_subscriptions: List[AlertSubscription],
        hours_since_last_check: int = 4
    ) -> List[Alert]:
        """
        Run incremental monitoring for new content and alerts.
        Used for real-time monitoring between comprehensive analyses.
        """
        
        async with async_performance_tracker("incremental_monitoring"):
            self.logger.info(f"Starting incremental monitoring for {len(competitors)} competitors")
            
            # Monitor recent content (last few hours)
            recent_content = await self._monitor_recent_content(
                competitors,
                hours_since_last_check
            )
            
            if not any(recent_content.values()):
                self.logger.info("No new content found during incremental monitoring")
                return []
            
            # Quick trend check on recent content
            all_recent_content = []
            for content_list in recent_content.values():
                all_recent_content.extend(content_list)
            
            # Use cached trends and gaps for alert evaluation
            cached_data = self._get_cached_analysis_results()
            cached_trends = cached_data.get("trends", [])
            cached_gaps = cached_data.get("gaps", [])
            cached_insights = cached_data.get("insights", [])
            
            # Check for alerts
            alerts = await self.alert_orchestrator.monitor_and_alert(
                competitors,
                recent_content,
                cached_trends,
                cached_gaps,
                cached_insights,
                alert_subscriptions
            )
            
            # Update workflow state
            self.workflow_state["last_incremental_analysis"] = datetime.utcnow()
            
            self.logger.info(f"Incremental monitoring completed. {len(alerts)} alerts generated")
            
            return alerts
    
    async def get_competitor_dashboard_data(
        self,
        competitors: List[Competitor],
        industry: Industry,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate dashboard data for competitive intelligence overview.
        Optimized for quick loading and regular updates.
        """
        
        # Check if we have recent cached data
        cached_data = self._get_cached_analysis_results()
        
        if cached_data and self._is_cache_valid():
            # Use cached data for dashboard
            dashboard_data = {
                "overview": {
                    "competitors_monitored": len(competitors),
                    "industry": industry.value,
                    "last_updated": cached_data.get("timestamp", datetime.utcnow()),
                    "analysis_status": "up_to_date"
                },
                "key_metrics": self._extract_key_metrics(cached_data),
                "top_trends": self._format_trends_for_dashboard(cached_data.get("trends", [])[:5]),
                "top_opportunities": self._format_gaps_for_dashboard(cached_data.get("gaps", [])[:5]),
                "competitor_performance": self._format_performance_for_dashboard(
                    cached_data.get("performance", {}),
                    competitors
                ),
                "recent_insights": self._format_insights_for_dashboard(cached_data.get("insights", [])[:10])
            }
        else:
            # Run lightweight analysis for dashboard
            dashboard_data = await self._generate_lightweight_dashboard(
                competitors,
                industry,
                time_range_days
            )
        
        return dashboard_data
    
    async def _orchestrate_content_monitoring(
        self,
        competitors: List[Competitor],
        analysis_depth: str
    ) -> Dict[str, List[ContentItem]]:
        """Orchestrate content monitoring across all competitors."""
        
        # Configure monitoring based on analysis depth
        monitoring_config = MonitoringConfig(
            check_frequency_hours=24,
            sentiment_analysis=analysis_depth in ["standard", "deep"],
            quality_scoring=analysis_depth in ["standard", "deep"],
            max_content_age_days=90 if analysis_depth == "deep" else 30
        )
        
        # Start monitoring session
        await self.content_monitor.start_monitoring_session()
        
        try:
            # Monitor competitors in batches
            batch_size = self.orchestration_config["analysis_batch_size"]
            competitor_content = {}
            
            for i in range(0, len(competitors), batch_size):
                batch = competitors[i:i + batch_size]
                
                # Monitor batch concurrently
                batch_tasks = [
                    self.content_monitor.monitor_competitor(competitor, monitoring_config)
                    for competitor in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for competitor, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to monitor {competitor.name}: {str(result)}")
                        competitor_content[competitor.id] = []
                    else:
                        competitor_content[competitor.id] = result
                        self.logger.info(f"Monitored {competitor.name}: {len(result)} content items")
        
        finally:
            # Close monitoring session
            await self.content_monitor.close_monitoring_session()
        
        return competitor_content
    
    async def _monitor_recent_content(
        self,
        competitors: List[Competitor],
        hours_since_last_check: int
    ) -> Dict[str, List[ContentItem]]:
        """Monitor only recent content for incremental updates."""
        
        # Lightweight monitoring config
        monitoring_config = MonitoringConfig(
            check_frequency_hours=hours_since_last_check,
            sentiment_analysis=False,  # Skip AI analysis for speed
            quality_scoring=False,
            max_content_age_days=1  # Only very recent content
        )
        
        await self.content_monitor.start_monitoring_session()
        
        try:
            # Monitor all competitors concurrently for recent content
            tasks = [
                self.content_monitor.monitor_competitor(competitor, monitoring_config)
                for competitor in competitors
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            recent_content = {}
            for competitor, result in zip(competitors, results):
                if isinstance(result, Exception):
                    self.logger.debug(f"Failed to monitor recent content for {competitor.name}: {str(result)}")
                    recent_content[competitor.id] = []
                else:
                    # Filter to only content from the specified time window
                    cutoff_time = datetime.utcnow() - timedelta(hours=hours_since_last_check)
                    filtered_content = [
                        item for item in result
                        if item.discovered_at >= cutoff_time
                    ]
                    recent_content[competitor.id] = filtered_content
        
        finally:
            await self.content_monitor.close_monitoring_session()
        
        return recent_content
    
    async def _generate_lightweight_dashboard(
        self,
        competitors: List[Competitor],
        industry: Industry,
        time_range_days: int
    ) -> Dict[str, Any]:
        """Generate lightweight dashboard data without full analysis."""
        
        # Run minimal content monitoring
        monitoring_config = MonitoringConfig(
            sentiment_analysis=False,
            quality_scoring=False,
            max_content_age_days=time_range_days
        )
        
        # Monitor subset of competitors for speed
        sample_competitors = competitors[:5]  # Monitor top 5 for dashboard
        
        await self.content_monitor.start_monitoring_session()
        
        try:
            content_data = {}
            for competitor in sample_competitors:
                content = await self.content_monitor.monitor_competitor(competitor, monitoring_config)
                content_data[competitor.id] = content
        finally:
            await self.content_monitor.close_monitoring_session()
        
        # Generate basic dashboard data
        total_content = sum(len(content) for content in content_data.values())
        
        dashboard_data = {
            "overview": {
                "competitors_monitored": len(competitors),
                "sample_analyzed": len(sample_competitors),
                "industry": industry.value,
                "last_updated": datetime.utcnow(),
                "analysis_status": "lightweight"
            },
            "key_metrics": {
                "total_content_found": total_content,
                "active_competitors": len([comp for comp, content in content_data.items() if content]),
                "avg_content_per_competitor": total_content / max(len(sample_competitors), 1)
            },
            "message": "Run comprehensive analysis for detailed insights"
        }
        
        return dashboard_data
    
    # Helper methods for data formatting
    
    def _extract_key_metrics(self, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from cached analysis data."""
        
        content_count = 0
        for content_list in cached_data.get("content", {}).values():
            content_count += len(content_list)
        
        return {
            "total_content_analyzed": content_count,
            "trends_identified": len(cached_data.get("trends", [])),
            "opportunities_found": len(cached_data.get("gaps", [])),
            "insights_generated": len(cached_data.get("insights", [])),
            "high_priority_opportunities": len([
                gap for gap in cached_data.get("gaps", [])
                if gap.opportunity_score > 80
            ])
        }
    
    def _format_trends_for_dashboard(self, trends: List[Trend]) -> List[Dict[str, Any]]:
        """Format trends for dashboard display."""
        
        return [
            {
                "topic": trend.topic,
                "strength": trend.strength.value,
                "growth_rate": f"{trend.growth_rate:.1%}",
                "opportunity_score": f"{trend.opportunity_score:.0f}%" if trend.opportunity_score else "N/A",
                "keywords": trend.keywords[:3]
            }
            for trend in trends
        ]
    
    def _format_gaps_for_dashboard(self, gaps: List[ContentGap]) -> List[Dict[str, Any]]:
        """Format content gaps for dashboard display."""
        
        return [
            {
                "topic": gap.topic,
                "opportunity_score": f"{gap.opportunity_score:.0f}%",
                "difficulty_score": f"{gap.difficulty_score:.0f}%",
                "potential_reach": f"{gap.potential_reach:,}",
                "missing_content_types": len(gap.content_types_missing)
            }
            for gap in gaps
        ]
    
    def _format_performance_for_dashboard(
        self,
        performance_data: Dict[str, Any],
        competitors: List[Competitor]
    ) -> List[Dict[str, Any]]:
        """Format performance data for dashboard display."""
        
        engagement_analysis = performance_data.get("engagement_analysis", {})
        
        competitor_performance = []
        for competitor in competitors[:10]:  # Top 10 for dashboard
            comp_data = engagement_analysis.get(competitor.id, {})
            
            competitor_performance.append({
                "name": competitor.name,
                "tier": competitor.tier.value,
                "avg_engagement": comp_data.get("avg_engagement", 0),
                "consistency": f"{comp_data.get('engagement_consistency', 0):.1%}",
                "trend": comp_data.get("engagement_trend", "stable"),
                "top_performers": comp_data.get("top_performers_count", 0)
            })
        
        # Sort by engagement
        competitor_performance.sort(key=lambda x: x["avg_engagement"], reverse=True)
        
        return competitor_performance
    
    def _format_insights_for_dashboard(self, insights: List[CompetitorInsight]) -> List[Dict[str, Any]]:
        """Format insights for dashboard display."""
        
        return [
            {
                "title": insight.title,
                "type": insight.insight_type,
                "impact": insight.impact_level,
                "confidence": f"{insight.confidence_score:.0%}",
                "description": insight.description[:200] + "..." if len(insight.description) > 200 else insight.description,
                "generated_at": insight.generated_at.strftime("%Y-%m-%d %H:%M")
            }
            for insight in insights
        ]
    
    # Cache management
    
    def _cache_analysis_results(self, results: Dict[str, Any]) -> None:
        """Cache analysis results for reuse."""
        
        results["timestamp"] = datetime.utcnow()
        self.workflow_state["cached_results"] = results
    
    def _get_cached_analysis_results(self) -> Dict[str, Any]:
        """Get cached analysis results."""
        
        return self.workflow_state.get("cached_results", {})
    
    def _is_cache_valid(self) -> bool:
        """Check if cached results are still valid."""
        
        cached_results = self.workflow_state.get("cached_results", {})
        if not cached_results:
            return False
        
        cache_time = cached_results.get("timestamp")
        if not cache_time:
            return False
        
        cache_age = datetime.utcnow() - cache_time
        max_age = timedelta(hours=self.orchestration_config["cache_duration_hours"])
        
        return cache_age < max_age
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health metrics."""
        
        return {
            "orchestrator_status": "active",
            "last_full_analysis": self.workflow_state.get("last_full_analysis"),
            "last_incremental_analysis": self.workflow_state.get("last_incremental_analysis"),
            "cache_status": "valid" if self._is_cache_valid() else "stale",
            "agent_status": {
                "content_monitor": await self.content_monitor.get_monitoring_status(),
                "trend_analyzer": "active",
                "gap_identifier": "active", 
                "performance_analyzer": "active",
                "strategic_insights": "active",
                "alert_orchestrator": await self.alert_orchestrator.get_alert_statistics(7)
            },
            "configuration": self.orchestration_config,
            "system_health": "healthy"
        }