"""
Comprehensive API usage analytics and monitoring system.
Tracks API calls, performance metrics, user behavior, and provides detailed insights.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import ipaddress

from fastapi import Request, Response
from pydantic import BaseModel
try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
    geoip2 = None

from .cache import cache
from .monitoring import metrics
from ..config.settings import settings

class RequestMethod(str, Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class APIAnalyticsEvent(str, Enum):
    """Types of API analytics events."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    RATE_LIMIT = "rate_limit"
    AUTH_FAILURE = "auth_failure"
    USAGE_THRESHOLD = "usage_threshold"

@dataclass
class APIRequestLog:
    """Individual API request log entry."""
    request_id: str
    timestamp: datetime
    method: RequestMethod
    path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    user_id: Optional[str]
    api_key_id: Optional[str]
    ip_address: str
    user_agent: str
    request_size: int
    response_status: int
    response_size: int
    response_time_ms: float
    api_version: str
    endpoint_category: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None

@dataclass 
class APIUsageMetrics:
    """Aggregated API usage metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    total_data_transferred_mb: float = 0.0
    unique_users: int = 0
    unique_ips: int = 0
    top_endpoints: Dict[str, int] = None
    top_user_agents: Dict[str, int] = None
    status_code_distribution: Dict[int, int] = None
    error_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.top_endpoints is None:
            self.top_endpoints = {}
        if self.top_user_agents is None:
            self.top_user_agents = {}
        if self.status_code_distribution is None:
            self.status_code_distribution = {}
        if self.error_distribution is None:
            self.error_distribution = {}

class APIAnalyticsCollector:
    """Collects and processes API analytics data."""
    
    def __init__(self, max_logs: int = 10000):
        self.request_logs: deque = deque(maxlen=max_logs)
        self.real_time_metrics: Dict[str, Any] = {}
        self.hourly_metrics: Dict[str, APIUsageMetrics] = {}
        self.daily_metrics: Dict[str, APIUsageMetrics] = {}
        self.user_metrics: Dict[str, APIUsageMetrics] = {}
        self.endpoint_metrics: Dict[str, APIUsageMetrics] = {}
        
        # Rate limiting tracking
        self.rate_limit_violations: deque = deque(maxlen=1000)
        self.suspicious_ips: Set[str] = set()
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        
        # GeoIP database (optional)
        self.geoip_reader = None
        if GEOIP_AVAILABLE:
            try:
                self.geoip_reader = geoip2.database.Reader('GeoLite2-City.mmdb')
            except:
                pass  # GeoIP is optional
    
    def _get_geo_location(self, ip_address: str) -> tuple[Optional[str], Optional[str]]:
        """Get geographic location from IP address."""
        if not self.geoip_reader:
            return None, None
        
        if GEOIP_AVAILABLE:
            try:
                response = self.geoip_reader.city(ip_address)
                return response.country.name, response.city.name
            except geoip2.errors.AddressNotFoundError:
                return None, None
            except:
                return None, None
        return None, None
    
    def _extract_endpoint_category(self, path: str) -> str:
        """Extract endpoint category from request path."""
        if path.startswith('/api/blogs'):
            return 'blogs'
        elif path.startswith('/api/campaigns'):
            return 'campaigns'
        elif path.startswith('/api/analytics'):
            return 'analytics'
        elif path.startswith('/api/webhooks'):
            return 'webhooks'
        elif path.startswith('/health'):
            return 'health'
        elif path.startswith('/docs'):
            return 'documentation'
        else:
            return 'other'
    
    def _is_suspicious_request(self, request_log: APIRequestLog) -> bool:
        """Detect potentially suspicious API requests."""
        # Check for high error rate from same IP
        recent_logs = [
            log for log in self.request_logs 
            if log.ip_address == request_log.ip_address 
            and log.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        if len(recent_logs) > 50:  # More than 50 requests in 5 minutes
            error_rate = sum(1 for log in recent_logs if not log.success) / len(recent_logs)
            if error_rate > 0.8:  # More than 80% errors
                return True
        
        # Check for unusual user agent patterns
        suspicious_user_agents = ['bot', 'crawler', 'scraper', 'scanner']
        if any(agent in request_log.user_agent.lower() for agent in suspicious_user_agents):
            return True
        
        return False
    
    async def log_request(
        self,
        request: Request,
        response: Response,
        response_time_ms: float,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None
    ):
        """Log an API request with comprehensive metadata."""
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        api_version = getattr(request.state, 'api_version', 'v2')
        
        # Calculate request/response sizes
        request_size = int(request.headers.get("content-length", 0))
        response_size = len(getattr(response, 'body', b'')) if hasattr(response, 'body') else 0
        
        # Get geographic location
        geo_country, geo_city = self._get_geo_location(client_ip)
        
        # Create request log
        request_log = APIRequestLog(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            method=RequestMethod(request.method),
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers),
            user_id=user_id,
            api_key_id=api_key_id,
            ip_address=client_ip,
            user_agent=user_agent,
            request_size=request_size,
            response_status=response.status_code,
            response_size=response_size,
            response_time_ms=response_time_ms,
            api_version=api_version,
            endpoint_category=self._extract_endpoint_category(request.url.path),
            success=200 <= response.status_code < 400,
            error_type=self._classify_error(response.status_code) if response.status_code >= 400 else None,
            geo_country=geo_country,
            geo_city=geo_city
        )
        
        # Store the log
        self.request_logs.append(request_log)
        
        # Update real-time metrics
        await self._update_real_time_metrics(request_log)
        
        # Update aggregated metrics
        await self._update_aggregated_metrics(request_log)
        
        # Check for suspicious activity
        if self._is_suspicious_request(request_log):
            self.suspicious_ips.add(client_ip)
            await self._alert_suspicious_activity(request_log)
        
        # Update performance metrics
        self.response_times.append(response_time_ms)
        if not request_log.success:
            self.error_counts[request_log.error_type] += 1
        
        # Store in cache for real-time dashboards
        await cache.set(
            "analytics",
            f"request_log:{request_log.request_id}",
            asdict(request_log),
            ttl=3600  # 1 hour
        )
    
    def _classify_error(self, status_code: int) -> str:
        """Classify error type based on HTTP status code."""
        if status_code == 400:
            return "BAD_REQUEST"
        elif status_code == 401:
            return "UNAUTHORIZED"
        elif status_code == 403:
            return "FORBIDDEN"
        elif status_code == 404:
            return "NOT_FOUND"
        elif status_code == 429:
            return "RATE_LIMITED"
        elif 400 <= status_code < 500:
            return "CLIENT_ERROR"
        elif 500 <= status_code < 600:
            return "SERVER_ERROR"
        else:
            return "UNKNOWN_ERROR"
    
    async def _update_real_time_metrics(self, request_log: APIRequestLog):
        """Update real-time metrics."""
        now = datetime.utcnow()
        current_minute = now.replace(second=0, microsecond=0)
        
        # Track requests per minute
        minute_key = current_minute.isoformat()
        if minute_key not in self.real_time_metrics:
            self.real_time_metrics[minute_key] = {
                'requests': 0,
                'errors': 0,
                'avg_response_time': 0,
                'data_transferred': 0
            }
        
        metrics_data = self.real_time_metrics[minute_key]
        metrics_data['requests'] += 1
        if not request_log.success:
            metrics_data['errors'] += 1
        
        # Update average response time
        old_avg = metrics_data['avg_response_time']
        new_avg = (old_avg * (metrics_data['requests'] - 1) + request_log.response_time_ms) / metrics_data['requests']
        metrics_data['avg_response_time'] = new_avg
        
        # Update data transferred
        metrics_data['data_transferred'] += (request_log.request_size + request_log.response_size) / (1024 * 1024)  # MB
        
        # Clean up old metrics (keep last 60 minutes)
        cutoff_time = now - timedelta(hours=1)
        self.real_time_metrics = {
            k: v for k, v in self.real_time_metrics.items()
            if datetime.fromisoformat(k) > cutoff_time
        }
    
    async def _update_aggregated_metrics(self, request_log: APIRequestLog):
        """Update hourly and daily aggregated metrics."""
        now = request_log.timestamp
        hour_key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        day_key = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Update hourly metrics
        if hour_key not in self.hourly_metrics:
            self.hourly_metrics[hour_key] = APIUsageMetrics()
        
        await self._update_usage_metrics(self.hourly_metrics[hour_key], request_log)
        
        # Update daily metrics
        if day_key not in self.daily_metrics:
            self.daily_metrics[day_key] = APIUsageMetrics()
        
        await self._update_usage_metrics(self.daily_metrics[day_key], request_log)
        
        # Update user-specific metrics
        if request_log.user_id:
            if request_log.user_id not in self.user_metrics:
                self.user_metrics[request_log.user_id] = APIUsageMetrics()
            await self._update_usage_metrics(self.user_metrics[request_log.user_id], request_log)
        
        # Update endpoint-specific metrics
        endpoint_key = f"{request_log.method}:{request_log.endpoint_category}"
        if endpoint_key not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint_key] = APIUsageMetrics()
        await self._update_usage_metrics(self.endpoint_metrics[endpoint_key], request_log)
    
    async def _update_usage_metrics(self, usage_metrics: APIUsageMetrics, request_log: APIRequestLog):
        """Update usage metrics with new request data."""
        usage_metrics.total_requests += 1
        
        if request_log.success:
            usage_metrics.successful_requests += 1
        else:
            usage_metrics.error_requests += 1
            usage_metrics.error_distribution[request_log.error_type] = (
                usage_metrics.error_distribution.get(request_log.error_type, 0) + 1
            )
        
        # Update response time metrics
        old_avg = usage_metrics.avg_response_time_ms
        usage_metrics.avg_response_time_ms = (
            (old_avg * (usage_metrics.total_requests - 1) + request_log.response_time_ms) / 
            usage_metrics.total_requests
        )
        
        # Update data transfer
        data_mb = (request_log.request_size + request_log.response_size) / (1024 * 1024)
        usage_metrics.total_data_transferred_mb += data_mb
        
        # Update distributions
        usage_metrics.status_code_distribution[request_log.response_status] = (
            usage_metrics.status_code_distribution.get(request_log.response_status, 0) + 1
        )
        
        usage_metrics.top_endpoints[request_log.path] = (
            usage_metrics.top_endpoints.get(request_log.path, 0) + 1
        )
        
        usage_metrics.top_user_agents[request_log.user_agent] = (
            usage_metrics.top_user_agents.get(request_log.user_agent, 0) + 1
        )
    
    async def _alert_suspicious_activity(self, request_log: APIRequestLog):
        """Alert about suspicious API activity."""
        from .webhooks import webhook_manager, WebhookEvent
        
        alert_data = {
            "type": "suspicious_activity",
            "ip_address": request_log.ip_address,
            "user_agent": request_log.user_agent,
            "endpoint": request_log.path,
            "timestamp": request_log.timestamp.isoformat(),
            "recent_error_rate": self._calculate_recent_error_rate(request_log.ip_address)
        }
        
        await webhook_manager.emit_event(
            WebhookEvent.SYSTEM_HEALTH,
            alert_data
        )
    
    def _calculate_recent_error_rate(self, ip_address: str) -> float:
        """Calculate recent error rate for an IP address."""
        recent_logs = [
            log for log in self.request_logs 
            if log.ip_address == ip_address 
            and log.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        if not recent_logs:
            return 0.0
        
        error_count = sum(1 for log in recent_logs if not log.success)
        return error_count / len(recent_logs)
    
    def get_real_time_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get real-time metrics for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        relevant_metrics = {
            k: v for k, v in self.real_time_metrics.items()
            if datetime.fromisoformat(k) > cutoff_time
        }
        
        # Calculate aggregated statistics
        total_requests = sum(m['requests'] for m in relevant_metrics.values())
        total_errors = sum(m['errors'] for m in relevant_metrics.values())
        avg_response_time = (
            sum(m['avg_response_time'] * m['requests'] for m in relevant_metrics.values()) / 
            total_requests if total_requests > 0 else 0
        )
        total_data = sum(m['data_transferred'] for m in relevant_metrics.values())
        
        return {
            "time_range_minutes": minutes,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "avg_response_time_ms": avg_response_time,
            "total_data_transferred_mb": total_data,
            "requests_per_minute": total_requests / minutes if minutes > 0 else 0,
            "timeline": relevant_metrics
        }
    
    def get_usage_analytics(
        self,
        period: str = "day",  # "hour", "day"
        limit: int = 24
    ) -> Dict[str, Any]:
        """Get usage analytics for a specific period."""
        
        if period == "hour":
            metrics_data = self.hourly_metrics
        else:
            metrics_data = self.daily_metrics
        
        # Get most recent metrics
        sorted_keys = sorted(metrics_data.keys(), reverse=True)[:limit]
        recent_metrics = {k: metrics_data[k] for k in sorted_keys}
        
        # Calculate overall statistics
        total_requests = sum(m.total_requests for m in recent_metrics.values())
        total_errors = sum(m.error_requests for m in recent_metrics.values())
        avg_response_time = (
            sum(m.avg_response_time_ms * m.total_requests for m in recent_metrics.values()) /
            total_requests if total_requests > 0 else 0
        )
        
        return {
            "period": period,
            "time_range": f"last_{limit}_{period}s",
            "summary": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / total_requests if total_requests > 0 else 0,
                "avg_response_time_ms": avg_response_time,
                "total_data_transferred_mb": sum(m.total_data_transferred_mb for m in recent_metrics.values())
            },
            "timeline": {
                k: asdict(v) for k, v in recent_metrics.items()
            }
        }
    
    def get_user_analytics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for a specific user."""
        if user_id not in self.user_metrics:
            return None
        
        user_metrics = self.user_metrics[user_id]
        
        # Get user's recent requests
        user_requests = [
            log for log in self.request_logs 
            if log.user_id == user_id 
            and log.timestamp > datetime.utcnow() - timedelta(days=30)
        ]
        
        return {
            "user_id": user_id,
            "metrics": asdict(user_metrics),
            "recent_activity": {
                "last_request": max(req.timestamp for req in user_requests).isoformat() if user_requests else None,
                "most_used_endpoints": dict(list(user_metrics.top_endpoints.items())[:10]),
                "geographic_distribution": self._get_user_geographic_distribution(user_requests)
            }
        }
    
    def _get_user_geographic_distribution(self, user_requests: List[APIRequestLog]) -> Dict[str, int]:
        """Get geographic distribution of user requests."""
        geo_distribution = defaultdict(int)
        for request in user_requests:
            if request.geo_country:
                geo_distribution[request.geo_country] += 1
        return dict(geo_distribution)
    
    def get_endpoint_analytics(self) -> Dict[str, Any]:
        """Get analytics for all endpoints."""
        return {
            "endpoints": {
                endpoint: asdict(metrics) 
                for endpoint, metrics in self.endpoint_metrics.items()
            },
            "performance_summary": {
                "slowest_endpoints": self._get_slowest_endpoints(),
                "highest_error_rate_endpoints": self._get_highest_error_rate_endpoints(),
                "most_popular_endpoints": self._get_most_popular_endpoints()
            }
        }
    
    def _get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with highest average response time."""
        endpoints = [
            {
                "endpoint": endpoint,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "total_requests": metrics.total_requests
            }
            for endpoint, metrics in self.endpoint_metrics.items()
            if metrics.total_requests > 10  # Only consider endpoints with meaningful traffic
        ]
        
        return sorted(endpoints, key=lambda x: x["avg_response_time_ms"], reverse=True)[:limit]
    
    def _get_highest_error_rate_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with highest error rates."""
        endpoints = [
            {
                "endpoint": endpoint,
                "error_rate": metrics.error_requests / metrics.total_requests,
                "total_requests": metrics.total_requests,
                "error_requests": metrics.error_requests
            }
            for endpoint, metrics in self.endpoint_metrics.items()
            if metrics.total_requests > 10  # Only consider endpoints with meaningful traffic
        ]
        
        return sorted(endpoints, key=lambda x: x["error_rate"], reverse=True)[:limit]
    
    def _get_most_popular_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular endpoints by request count."""
        endpoints = [
            {
                "endpoint": endpoint,
                "total_requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / metrics.total_requests,
                "avg_response_time_ms": metrics.avg_response_time_ms
            }
            for endpoint, metrics in self.endpoint_metrics.items()
        ]
        
        return sorted(endpoints, key=lambda x: x["total_requests"], reverse=True)[:limit]
    
    def get_security_analytics(self) -> Dict[str, Any]:
        """Get security-related analytics."""
        recent_time = datetime.utcnow() - timedelta(hours=24)
        recent_logs = [log for log in self.request_logs if log.timestamp > recent_time]
        
        # Calculate security metrics
        auth_failures = [log for log in recent_logs if log.error_type == "UNAUTHORIZED"]
        rate_limit_violations = [log for log in recent_logs if log.error_type == "RATE_LIMITED"]
        
        # IP analysis
        ip_distribution = defaultdict(int)
        for log in recent_logs:
            ip_distribution[log.ip_address] += 1
        
        suspicious_ips = [
            ip for ip, count in ip_distribution.items() 
            if count > 1000  # More than 1000 requests in 24 hours
        ]
        
        return {
            "timeframe": "last_24_hours",
            "auth_failures": len(auth_failures),
            "rate_limit_violations": len(rate_limit_violations),
            "suspicious_ips": list(self.suspicious_ips),
            "high_volume_ips": suspicious_ips,
            "geographic_distribution": self._get_geographic_distribution(recent_logs),
            "user_agent_analysis": self._analyze_user_agents(recent_logs)
        }
    
    def _get_geographic_distribution(self, logs: List[APIRequestLog]) -> Dict[str, int]:
        """Get geographic distribution of requests."""
        geo_distribution = defaultdict(int)
        for log in logs:
            if log.geo_country:
                geo_distribution[log.geo_country] += 1
        return dict(geo_distribution)
    
    def _analyze_user_agents(self, logs: List[APIRequestLog]) -> Dict[str, Any]:
        """Analyze user agent patterns."""
        user_agents = [log.user_agent for log in logs]
        agent_counts = defaultdict(int)
        
        for agent in user_agents:
            agent_counts[agent] += 1
        
        # Categorize user agents
        bot_agents = [agent for agent in agent_counts.keys() if 'bot' in agent.lower()]
        browser_agents = [agent for agent in agent_counts.keys() if any(browser in agent.lower() for browser in ['chrome', 'firefox', 'safari', 'edge'])]
        
        return {
            "total_unique_agents": len(agent_counts),
            "bot_agents": len(bot_agents),
            "browser_agents": len(browser_agents),
            "top_agents": dict(list(sorted(agent_counts.items(), key=lambda x: x[1], reverse=True))[:10])
        }

# Global analytics collector instance
analytics_collector = APIAnalyticsCollector()

class APIAnalyticsMiddleware:
    """Middleware to collect API analytics automatically."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Create response wrapper to capture response data
        response_data = {}
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_data["status_code"] = message["status"]
                response_data["headers"] = dict(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_data.setdefault("body", b"")
                response_data["body"] += message.get("body", b"")
            
            await send(message)
        
        # Process request
        await self.app(scope, receive, send_wrapper)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Create mock response object for analytics
        class MockResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self.body = body
        
        mock_response = MockResponse(
            response_data.get("status_code", 200),
            response_data.get("body", b"")
        )
        
        # Extract user information (if available)
        user_id = getattr(request.state, 'user_id', None)
        api_key_id = getattr(request.state, 'api_key_id', None)
        
        # Log the request asynchronously
        asyncio.create_task(
            analytics_collector.log_request(
                request=request,
                response=mock_response,
                response_time_ms=response_time_ms,
                user_id=user_id,
                api_key_id=api_key_id
            )
        )