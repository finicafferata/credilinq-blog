"""
Performance Monitoring Service for Parallel Research Implementation

This service provides comprehensive monitoring and validation of parallel research
performance, specifically tracking the 40% time reduction target from User Story 2.1.

Key Features:
- Real-time performance tracking during parallel execution
- Validation of time reduction targets
- Quality preservation monitoring
- Performance degradation alerts
- Historical performance analysis
- Automated reporting and recommendations
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import deque, defaultdict
import json

# Performance tracking infrastructure
try:
    from ...core.langgraph_performance_tracker import global_performance_tracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceStatus(Enum):
    """Performance status levels."""
    EXCELLENT = "excellent"      # > 110% of target
    GOOD = "good"               # 90-110% of target
    ACCEPTABLE = "acceptable"   # 70-90% of target
    POOR = "poor"              # 50-70% of target
    CRITICAL = "critical"      # < 50% of target


class AlertType(Enum):
    """Types of performance alerts."""
    TIME_REDUCTION_BELOW_TARGET = "time_reduction_below_target"
    QUALITY_DEGRADATION = "quality_degradation"
    AGENT_FAILURE = "agent_failure"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    EFFICIENCY_DECLINE = "efficiency_decline"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for parallel research."""
    # Timing metrics
    sequential_time_estimate: float = 0.0
    parallel_execution_time: float = 0.0
    time_reduction_percentage: float = 0.0
    parallel_efficiency_gain: float = 0.0
    
    # Quality metrics
    research_completeness_score: float = 0.0
    quality_preservation_score: float = 0.0
    source_quality_average: float = 0.0
    insight_confidence_average: float = 0.0
    conflict_resolution_rate: float = 0.0
    
    # Agent performance
    researcher_execution_time: float = 0.0
    search_agent_execution_time: float = 0.0
    researcher_success_rate: float = 0.0
    search_agent_success_rate: float = 0.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    
    # Result metrics
    sources_processed: int = 0
    insights_generated: int = 0
    conflicts_resolved: int = 0
    duplicates_removed: int = 0
    
    # Error tracking
    error_count: int = 0
    timeout_count: int = 0
    retry_count: int = 0
    
    # Metadata
    workflow_id: str = ""
    execution_timestamp: datetime = field(default_factory=datetime.now)
    research_topics: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert with severity and recommendations."""
    alert_type: AlertType
    severity: PerformanceStatus
    message: str
    current_value: float
    target_value: float
    recommendations: List[str]
    workflow_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceTrend:
    """Historical performance trend analysis."""
    metric_name: str
    current_value: float
    trend_direction: str  # "improving", "declining", "stable"
    change_percentage: float
    historical_average: float
    confidence_interval: Tuple[float, float]
    data_points: int


class ParallelResearchPerformanceMonitor:
    """Comprehensive performance monitor for parallel research implementation."""
    
    def __init__(self, 
                 time_reduction_target: float = 0.40,
                 quality_preservation_target: float = 0.95,
                 history_size: int = 100):
        """Initialize the performance monitor."""
        self.time_reduction_target = time_reduction_target
        self.quality_preservation_target = quality_preservation_target
        
        # Performance history
        self.performance_history = deque(maxlen=history_size)
        self.metric_trends = {}
        
        # Alert system
        self.active_alerts = []
        self.alert_thresholds = {
            AlertType.TIME_REDUCTION_BELOW_TARGET: time_reduction_target * 0.8,  # 32% for 40% target
            AlertType.QUALITY_DEGRADATION: quality_preservation_target * 0.9,   # 85.5% for 95% target
            AlertType.EFFICIENCY_DECLINE: 1.2  # Efficiency should be > 1.2x
        }
        
        # Performance baselines
        self.baseline_metrics = None
        self.target_achievements = defaultdict(int)
        self.total_executions = 0
        
        logger.info(f"ParallelResearchPerformanceMonitor initialized")
        logger.info(f"Targets: {time_reduction_target*100}% time reduction, {quality_preservation_target*100}% quality preservation")
    
    async def start_monitoring(self, workflow_id: str, research_topics: List[str]) -> str:
        """Start monitoring a parallel research execution."""
        monitoring_session_id = f"monitor_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting performance monitoring for workflow: {workflow_id}")
        logger.info(f"Monitoring session: {monitoring_session_id}")
        
        # Initialize baseline metrics
        baseline_metrics = PerformanceMetrics(
            workflow_id=workflow_id,
            research_topics=research_topics,
            execution_timestamp=datetime.now()
        )
        
        # Track with global performance tracker if available
        if TRACKING_AVAILABLE:
            await global_performance_tracker.track_workflow_start(
                workflow_id=monitoring_session_id,
                workflow_type="parallel_research_monitoring",
                metadata={
                    "original_workflow_id": workflow_id,
                    "research_topics": research_topics,
                    "time_reduction_target": self.time_reduction_target,
                    "quality_preservation_target": self.quality_preservation_target
                }
            )
        
        return monitoring_session_id
    
    async def record_agent_start(self, 
                                 workflow_id: str, 
                                 agent_name: str,
                                 expected_duration: Optional[float] = None) -> datetime:
        """Record the start of an agent execution."""
        start_time = datetime.now()
        
        logger.info(f"Agent {agent_name} started for workflow {workflow_id}")
        if expected_duration:
            logger.info(f"Expected duration: {expected_duration:.2f}s")
        
        return start_time
    
    async def record_agent_completion(self,
                                     workflow_id: str,
                                     agent_name: str,
                                     start_time: datetime,
                                     success: bool,
                                     error_message: Optional[str] = None) -> float:
        """Record the completion of an agent execution."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Agent {agent_name} completed for workflow {workflow_id}")
        logger.info(f"Execution time: {execution_time:.2f}s, Success: {success}")
        
        if not success and error_message:
            logger.warning(f"Agent {agent_name} failed: {error_message}")
        
        return execution_time
    
    async def record_parallel_execution(self,
                                       workflow_id: str,
                                       researcher_time: float,
                                       search_agent_time: float,
                                       parallel_start: datetime,
                                       parallel_end: datetime,
                                       researcher_success: bool,
                                       search_agent_success: bool) -> PerformanceMetrics:
        """Record parallel execution performance."""
        
        # Calculate parallel execution metrics
        parallel_time = (parallel_end - parallel_start).total_seconds()
        sequential_time_estimate = researcher_time + search_agent_time
        
        if sequential_time_estimate > 0:
            time_reduction_percentage = ((sequential_time_estimate - parallel_time) / sequential_time_estimate) * 100
            parallel_efficiency_gain = sequential_time_estimate / parallel_time if parallel_time > 0 else 1.0
        else:
            time_reduction_percentage = 0.0
            parallel_efficiency_gain = 1.0
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            workflow_id=workflow_id,
            sequential_time_estimate=sequential_time_estimate,
            parallel_execution_time=parallel_time,
            time_reduction_percentage=time_reduction_percentage,
            parallel_efficiency_gain=parallel_efficiency_gain,
            researcher_execution_time=researcher_time,
            search_agent_execution_time=search_agent_time,
            researcher_success_rate=1.0 if researcher_success else 0.0,
            search_agent_success_rate=1.0 if search_agent_success else 0.0,
            execution_timestamp=parallel_start
        )
        
        # Log performance results
        logger.info(f"=== Parallel Execution Performance ===")
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Sequential estimate: {sequential_time_estimate:.2f}s")
        logger.info(f"Parallel execution: {parallel_time:.2f}s")
        logger.info(f"Time reduction: {time_reduction_percentage:.1f}%")
        logger.info(f"Efficiency gain: {parallel_efficiency_gain:.2f}x")
        logger.info(f"Target achievement: {time_reduction_percentage >= (self.time_reduction_target * 100)}")
        
        # Check for alerts
        await self._check_performance_alerts(metrics)
        
        # Update tracking statistics
        self.total_executions += 1
        if time_reduction_percentage >= (self.time_reduction_target * 100):
            self.target_achievements['time_reduction'] += 1
        
        # Add to history
        self.performance_history.append(metrics)
        
        # Update trends
        await self._update_performance_trends(metrics)
        
        return metrics
    
    async def record_quality_metrics(self,
                                    workflow_id: str,
                                    research_completeness: float,
                                    quality_preservation: float,
                                    source_quality_avg: float,
                                    insight_confidence_avg: float,
                                    conflict_resolution_rate: float,
                                    sources_processed: int,
                                    insights_generated: int,
                                    conflicts_resolved: int) -> None:
        """Record quality metrics for the parallel research execution."""
        
        # Find the latest metrics entry for this workflow
        latest_metrics = None
        for metrics in reversed(self.performance_history):
            if metrics.workflow_id == workflow_id:
                latest_metrics = metrics
                break
        
        if latest_metrics:
            # Update quality metrics
            latest_metrics.research_completeness_score = research_completeness
            latest_metrics.quality_preservation_score = quality_preservation
            latest_metrics.source_quality_average = source_quality_avg
            latest_metrics.insight_confidence_average = insight_confidence_avg
            latest_metrics.conflict_resolution_rate = conflict_resolution_rate
            latest_metrics.sources_processed = sources_processed
            latest_metrics.insights_generated = insights_generated
            latest_metrics.conflicts_resolved = conflicts_resolved
            
            logger.info(f"=== Quality Metrics Updated ===")
            logger.info(f"Research completeness: {research_completeness:.1%}")
            logger.info(f"Quality preservation: {quality_preservation:.1%}")
            logger.info(f"Source quality average: {source_quality_avg:.2f}")
            logger.info(f"Insight confidence average: {insight_confidence_avg:.2f}")
            logger.info(f"Conflict resolution rate: {conflict_resolution_rate:.1%}")
            
            # Check quality alerts
            if quality_preservation < self.quality_preservation_target:
                await self._generate_quality_alert(workflow_id, quality_preservation)
            
            # Update quality target achievement
            if quality_preservation >= self.quality_preservation_target:
                self.target_achievements['quality_preservation'] += 1
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts based on current metrics."""
        alerts = []
        
        # Time reduction alert
        if metrics.time_reduction_percentage < (self.time_reduction_target * 100):
            alert = PerformanceAlert(
                alert_type=AlertType.TIME_REDUCTION_BELOW_TARGET,
                severity=self._calculate_severity(
                    metrics.time_reduction_percentage / 100,
                    self.time_reduction_target
                ),
                message=f"Time reduction {metrics.time_reduction_percentage:.1f}% below target {self.time_reduction_target * 100}%",
                current_value=metrics.time_reduction_percentage,
                target_value=self.time_reduction_target * 100,
                recommendations=self._get_time_reduction_recommendations(metrics),
                workflow_id=metrics.workflow_id
            )
            alerts.append(alert)
        
        # Efficiency alert
        if metrics.parallel_efficiency_gain < self.alert_thresholds[AlertType.EFFICIENCY_DECLINE]:
            alert = PerformanceAlert(
                alert_type=AlertType.EFFICIENCY_DECLINE,
                severity=PerformanceStatus.POOR,
                message=f"Parallel efficiency {metrics.parallel_efficiency_gain:.2f}x below expected threshold",
                current_value=metrics.parallel_efficiency_gain,
                target_value=self.alert_thresholds[AlertType.EFFICIENCY_DECLINE],
                recommendations=[
                    "Check agent execution balance",
                    "Optimize slower agent performance",
                    "Consider timeout adjustments"
                ],
                workflow_id=metrics.workflow_id
            )
            alerts.append(alert)
        
        # Agent failure alerts
        if metrics.researcher_success_rate < 1.0:
            alert = PerformanceAlert(
                alert_type=AlertType.AGENT_FAILURE,
                severity=PerformanceStatus.CRITICAL,
                message="ResearcherAgent execution failed",
                current_value=metrics.researcher_success_rate,
                target_value=1.0,
                recommendations=[
                    "Check ResearcherAgent configuration",
                    "Verify data source availability",
                    "Review error logs for root cause"
                ],
                workflow_id=metrics.workflow_id
            )
            alerts.append(alert)
        
        if metrics.search_agent_success_rate < 1.0:
            alert = PerformanceAlert(
                alert_type=AlertType.AGENT_FAILURE,
                severity=PerformanceStatus.CRITICAL,
                message="SearchAgent execution failed",
                current_value=metrics.search_agent_success_rate,
                target_value=1.0,
                recommendations=[
                    "Check SearchAgent configuration",
                    "Verify search service connectivity",
                    "Review error logs for root cause"
                ],
                workflow_id=metrics.workflow_id
            )
            alerts.append(alert)
        
        # Add alerts and log them
        for alert in alerts:
            self.active_alerts.append(alert)
            logger.warning(f"PERFORMANCE ALERT: {alert.alert_type.value} - {alert.message}")
            for rec in alert.recommendations:
                logger.warning(f"  RECOMMENDATION: {rec}")
    
    async def _generate_quality_alert(self, workflow_id: str, quality_preservation: float) -> None:
        """Generate quality preservation alert."""
        alert = PerformanceAlert(
            alert_type=AlertType.QUALITY_DEGRADATION,
            severity=self._calculate_severity(quality_preservation, self.quality_preservation_target),
            message=f"Quality preservation {quality_preservation:.1%} below target {self.quality_preservation_target:.1%}",
            current_value=quality_preservation,
            target_value=self.quality_preservation_target,
            recommendations=[
                "Review source quality filtering",
                "Adjust conflict resolution strategies",
                "Increase source diversity requirements",
                "Enhance insight validation logic"
            ],
            workflow_id=workflow_id
        )
        
        self.active_alerts.append(alert)
        logger.warning(f"QUALITY ALERT: {alert.message}")
        for rec in alert.recommendations:
            logger.warning(f"  RECOMMENDATION: {rec}")
    
    def _calculate_severity(self, current_value: float, target_value: float) -> PerformanceStatus:
        """Calculate alert severity based on deviation from target."""
        ratio = current_value / target_value if target_value > 0 else 0
        
        if ratio >= 1.1:
            return PerformanceStatus.EXCELLENT
        elif ratio >= 0.9:
            return PerformanceStatus.GOOD
        elif ratio >= 0.7:
            return PerformanceStatus.ACCEPTABLE
        elif ratio >= 0.5:
            return PerformanceStatus.POOR
        else:
            return PerformanceStatus.CRITICAL
    
    def _get_time_reduction_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get specific recommendations for improving time reduction."""
        recommendations = []
        
        # Analyze execution time imbalance
        time_difference = abs(metrics.researcher_execution_time - metrics.search_agent_execution_time)
        if time_difference > max(metrics.researcher_execution_time, metrics.search_agent_execution_time) * 0.5:
            recommendations.append("Significant execution time imbalance detected - optimize slower agent")
        
        # Agent-specific recommendations
        if metrics.researcher_execution_time > metrics.search_agent_execution_time * 1.5:
            recommendations.extend([
                "ResearcherAgent is bottleneck - reduce research depth or scope",
                "Consider caching frequently researched topics",
                "Optimize source validation algorithms"
            ])
        elif metrics.search_agent_execution_time > metrics.researcher_execution_time * 1.5:
            recommendations.extend([
                "SearchAgent is bottleneck - reduce max sources per agent",
                "Optimize search query generation",
                "Consider parallel search execution within agent"
            ])
        
        # General recommendations
        if metrics.parallel_efficiency_gain < 1.3:
            recommendations.extend([
                "Consider agent timeout optimization",
                "Review concurrent execution limits",
                "Analyze resource contention"
            ])
        
        return recommendations
    
    async def _update_performance_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trend analysis."""
        if len(self.performance_history) < 2:
            return
        
        # Calculate trends for key metrics
        recent_history = list(self.performance_history)[-10:]  # Last 10 executions
        
        trend_metrics = {
            'time_reduction_percentage': [m.time_reduction_percentage for m in recent_history],
            'parallel_efficiency_gain': [m.parallel_efficiency_gain for m in recent_history],
            'quality_preservation_score': [m.quality_preservation_score for m in recent_history if m.quality_preservation_score > 0],
            'research_completeness_score': [m.research_completeness_score for m in recent_history if m.research_completeness_score > 0]
        }
        
        for metric_name, values in trend_metrics.items():
            if len(values) >= 3:  # Need at least 3 data points for trend
                trend = self._calculate_trend(values)
                self.metric_trends[metric_name] = trend
    
    def _calculate_trend(self, values: List[float]) -> PerformanceTrend:
        """Calculate trend for a series of values."""
        if len(values) < 2:
            return PerformanceTrend(
                metric_name="unknown",
                current_value=values[0] if values else 0,
                trend_direction="stable",
                change_percentage=0,
                historical_average=values[0] if values else 0,
                confidence_interval=(0, 0),
                data_points=len(values)
            )
        
        current_value = values[-1]
        historical_average = statistics.mean(values[:-1])
        change_percentage = ((current_value - historical_average) / historical_average * 100) if historical_average > 0 else 0
        
        # Determine trend direction
        if change_percentage > 5:
            trend_direction = "improving"
        elif change_percentage < -5:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        # Calculate confidence interval (simplified)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        confidence_interval = (
            historical_average - 1.96 * std_dev / len(values) ** 0.5,
            historical_average + 1.96 * std_dev / len(values) ** 0.5
        )
        
        return PerformanceTrend(
            metric_name="",
            current_value=current_value,
            trend_direction=trend_direction,
            change_percentage=change_percentage,
            historical_average=historical_average,
            confidence_interval=confidence_interval,
            data_points=len(values)
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_metrics = list(self.performance_history)[-10:]  # Last 10 executions
        
        summary = {
            "overall_statistics": {
                "total_executions": self.total_executions,
                "time_reduction_target_achievement_rate": (
                    self.target_achievements['time_reduction'] / self.total_executions
                    if self.total_executions > 0 else 0
                ),
                "quality_preservation_target_achievement_rate": (
                    self.target_achievements['quality_preservation'] / self.total_executions
                    if self.total_executions > 0 else 0
                )
            },
            "recent_performance": {
                "average_time_reduction": statistics.mean(
                    [m.time_reduction_percentage for m in recent_metrics]
                ),
                "average_efficiency_gain": statistics.mean(
                    [m.parallel_efficiency_gain for m in recent_metrics]
                ),
                "average_quality_preservation": statistics.mean(
                    [m.quality_preservation_score for m in recent_metrics if m.quality_preservation_score > 0]
                ) if any(m.quality_preservation_score > 0 for m in recent_metrics) else 0,
                "success_rate": statistics.mean(
                    [min(m.researcher_success_rate, m.search_agent_success_rate) for m in recent_metrics]
                )
            },
            "performance_trends": {
                metric: {
                    "direction": trend.trend_direction,
                    "change_percentage": trend.change_percentage,
                    "current_value": trend.current_value,
                    "historical_average": trend.historical_average
                }
                for metric, trend in self.metric_trends.items()
            },
            "active_alerts": [
                {
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "recommendations": alert.recommendations,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts[-5:]  # Last 5 alerts
            ],
            "targets": {
                "time_reduction_target": self.time_reduction_target * 100,
                "quality_preservation_target": self.quality_preservation_target * 100
            }
        }
        
        return summary
    
    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        summary = self.get_performance_summary()
        
        if "message" in summary:
            return summary["message"]
        
        report_lines = [
            "=== Parallel Research Performance Report ===",
            "",
            f"Total Executions: {summary['overall_statistics']['total_executions']}",
            f"Time Reduction Target Achievement: {summary['overall_statistics']['time_reduction_target_achievement_rate']:.1%}",
            f"Quality Preservation Target Achievement: {summary['overall_statistics']['quality_preservation_target_achievement_rate']:.1%}",
            "",
            "=== Recent Performance (Last 10 Executions) ===",
            f"Average Time Reduction: {summary['recent_performance']['average_time_reduction']:.1f}%",
            f"Average Efficiency Gain: {summary['recent_performance']['average_efficiency_gain']:.2f}x",
            f"Average Quality Preservation: {summary['recent_performance']['average_quality_preservation']:.1%}",
            f"Success Rate: {summary['recent_performance']['success_rate']:.1%}",
            "",
            "=== Performance Trends ===",
        ]
        
        for metric, trend_data in summary['performance_trends'].items():
            report_lines.append(
                f"{metric}: {trend_data['direction']} "
                f"({trend_data['change_percentage']:+.1f}% from historical average)"
            )
        
        if summary['active_alerts']:
            report_lines.extend([
                "",
                "=== Active Alerts ===",
            ])
            for alert in summary['active_alerts']:
                report_lines.append(f"- {alert['severity'].upper()}: {alert['message']}")
        
        report_lines.extend([
            "",
            "=== Targets ===",
            f"Time Reduction Target: {summary['targets']['time_reduction_target']}%",
            f"Quality Preservation Target: {summary['targets']['quality_preservation_target']}%"
        ])
        
        return "\n".join(report_lines)
    
    def clear_old_alerts(self, hours: int = 24) -> None:
        """Clear alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if alert.timestamp > cutoff_time
        ]
    
    async def export_metrics(self, filepath: str) -> None:
        """Export performance metrics to JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_history": [
                {
                    "workflow_id": m.workflow_id,
                    "execution_timestamp": m.execution_timestamp.isoformat(),
                    "time_reduction_percentage": m.time_reduction_percentage,
                    "parallel_efficiency_gain": m.parallel_efficiency_gain,
                    "quality_preservation_score": m.quality_preservation_score,
                    "research_completeness_score": m.research_completeness_score,
                    "sources_processed": m.sources_processed,
                    "insights_generated": m.insights_generated,
                    "researcher_execution_time": m.researcher_execution_time,
                    "search_agent_execution_time": m.search_agent_execution_time,
                    "researcher_success_rate": m.researcher_success_rate,
                    "search_agent_success_rate": m.search_agent_success_rate
                }
                for m in self.performance_history
            ],
            "summary": self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global performance monitor instance
parallel_research_performance_monitor = ParallelResearchPerformanceMonitor()

logger.info("🔍 Parallel Research Performance Monitor loaded successfully!")