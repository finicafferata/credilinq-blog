# Review Workflow - Technical Architecture

## Overview
This document provides the complete technical architecture for implementing the 4-stage Review Workflow system within the existing LangGraph-based CrediLinq Content Agent platform.

---

## **LangGraph Workflow Design**

### **Architecture Decision: Separate Workflow with Integration**
The Review Workflow is implemented as a **separate LangGraph workflow** that integrates seamlessly with existing content generation workflows. This design provides:

- Clean separation of concerns between generation and review
- Reusability across different content types
- Independent scaling and performance optimization
- Easy maintenance and updates

### **Workflow Node Structure**

```python
# Core workflow nodes
review_workflow_nodes = {
    # Automated agent nodes
    "quality_check": QualityCheckNode,
    "brand_check": BrandCheckNode, 
    "seo_review": SEOReviewNode,
    "final_approval": FinalApprovalNode,
    
    # Human review checkpoint nodes
    "quality_human_review": HumanReviewCheckpointNode,
    "brand_human_review": HumanReviewCheckpointNode,
    "seo_human_review": HumanReviewCheckpointNode,
    "approval_human_review": HumanReviewCheckpointNode,
    
    # Control flow nodes
    "workflow_complete": CompletionNode,
    "handle_rejection": RejectionHandlerNode,
}
```

### **State Management Pattern**

```python
@dataclass
class ReviewWorkflowState:
    """LangGraph-compatible state for review workflow"""
    content_id: str
    content_type: str
    content_data: Dict[str, Any]
    campaign_id: Optional[str] = None
    
    # Stage tracking
    current_stage: ReviewStage = ReviewStage.QUALITY_CHECK
    completed_stages: List[ReviewStage] = field(default_factory=list)
    failed_stages: List[ReviewStage] = field(default_factory=list)
    
    # Human review management
    active_checkpoints: Dict[str, ReviewCheckpoint] = field(default_factory=dict)
    review_history: List[ReviewFeedback] = field(default_factory=list)
    
    # Workflow control
    is_paused: bool = False
    workflow_status: ReviewStatus = ReviewStatus.PENDING
    
    # Configuration
    auto_approve_threshold: float = 0.85
    require_human_approval: bool = True
    parallel_reviews: bool = False
```

---

## **Human-in-the-Loop Implementation**

### **Pause/Resume Pattern**

The workflow uses a **persistent checkpoint system** that allows pausing at any human review point and resuming when feedback is received.

```python
async def human_review_checkpoint_node(self, state: ReviewWorkflowState) -> ReviewWorkflowState:
    """Human review checkpoint with pause/resume functionality"""
    current_stage = state.current_stage
    
    # Check for pending human feedback
    pending_feedback = await self._get_pending_human_feedback(state.content_id, current_stage)
    
    if pending_feedback:
        # Process feedback and continue workflow
        state.review_history.append(pending_feedback)
        self._update_workflow_status(state, pending_feedback)
    else:
        # No feedback yet - pause workflow and notify reviewer
        await self._ensure_reviewer_notified(state, current_stage)
        state.is_paused = True
        state.pause_reason = f"Waiting for human review: {current_stage.value}"
    
    return state
```

### **Resume Workflow Mechanism**

```python
async def resume_review_workflow(self, content_id: str) -> Dict[str, Any]:
    """Resume paused workflow when human feedback received"""
    # Load persisted workflow state
    workflow_state = await self.state_manager.load_workflow_state(f"review_{content_id}")
    
    # Check for new human feedback
    await self._refresh_human_feedback(workflow_state)
    
    if workflow_state.is_paused and self._has_new_feedback(workflow_state):
        # Resume workflow execution from current state
        workflow_state.is_paused = False
        result = await self.graph.ainvoke(
            workflow_state,
            config={"configurable": {"thread_id": f"review_{content_id}"}}
        )
        
        return {"status": "resumed", "current_stage": result.current_stage.value}
    
    return {"status": "waiting_for_feedback"}
```

---

## **Database Schema Design**

### **Core Review Tables**

```sql
-- Main workflow tracking table
CREATE TABLE content_review_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL,
    campaign_id UUID NULL,
    workflow_execution_id VARCHAR(255) NOT NULL UNIQUE,
    current_stage VARCHAR(50) NOT NULL,
    workflow_status VARCHAR(50) NOT NULL,
    state_data JSONB NOT NULL,
    is_paused BOOLEAN DEFAULT FALSE,
    pause_reason TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

-- Human review checkpoints
CREATE TABLE review_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL,
    workflow_execution_id VARCHAR(255) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    reviewer_id VARCHAR(255) NULL,
    assigned_at TIMESTAMPTZ NULL,
    deadline TIMESTAMPTZ NULL,
    status VARCHAR(50) DEFAULT 'pending',
    automated_score DECIMAL(3,2) NULL,
    automated_feedback JSONB NULL,
    human_feedback JSONB NULL,
    requires_human BOOLEAN DEFAULT FALSE,
    notification_sent BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMPTZ NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simple review feedback (no compliance overhead)
CREATE TABLE review_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL,
    reviewer_id VARCHAR(255) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    decision VARCHAR(50) NOT NULL, -- approve, reject, request_changes
    comments TEXT DEFAULT '',
    reviewed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Notification tracking
CREATE TABLE review_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    checkpoint_id UUID NOT NULL REFERENCES review_checkpoints(id) ON DELETE CASCADE,
    reviewer_id VARCHAR(255) NOT NULL,
    notification_type VARCHAR(50) NOT NULL, -- email, slack, dashboard
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ NULL,
    delivery_status VARCHAR(50) DEFAULT 'sent',
    metadata JSONB DEFAULT '{}'
);

-- Performance indexes
CREATE INDEX idx_review_workflows_content_id ON content_review_workflows(content_id);
CREATE INDEX idx_review_workflows_status ON content_review_workflows(workflow_status);
CREATE INDEX idx_review_checkpoints_content_stage ON review_checkpoints(content_id, stage);
CREATE INDEX idx_review_checkpoints_reviewer_status ON review_checkpoints(reviewer_id, status);
CREATE INDEX idx_review_feedback_reviewer_stage ON review_feedback(reviewer_id, stage);
```

---

## **AI Review Agents Integration**

### **BaseAgent Integration Pattern**

All review agents extend the existing `BaseAgent` architecture for consistency:

```python
class QualityCheckAgent(BaseAgent):
    """AI agent for content quality validation"""
    
    def __init__(self):
        super().__init__(
            agent_name="QualityCheckAgent",
            agent_type="quality_reviewer",
            capabilities=[
                "accuracy_validation",
                "structure_analysis", 
                "relevance_scoring",
                "readability_assessment"
            ]
        )
        
    async def execute_safe(self, content_data: Dict[str, Any], context: AgentExecutionContext) -> AgentResult:
        """Execute quality check with performance tracking"""
        try:
            # Perform comprehensive quality analysis
            quality_analysis = await self._analyze_content_quality(content_data)
            
            # Generate confidence score and recommendations
            confidence_score = self._calculate_confidence_score(quality_analysis)
            recommendations = self._generate_quality_recommendations(quality_analysis)
            
            # Determine if human review required
            requires_human_review = confidence_score < self.auto_approve_threshold
            
            return AgentResult(
                is_success=True,
                data={
                    "quality_analysis": quality_analysis,
                    "confidence_score": confidence_score,
                    "requires_human_review": requires_human_review,
                    "recommendations": recommendations,
                    "auto_approved": not requires_human_review
                },
                agent_reasoning=f"Quality analysis completed with {confidence_score:.2f} confidence",
                performance_metrics=self.get_performance_metrics()
            )
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return AgentResult(
                is_success=False,
                error_message=str(e),
                agent_reasoning="Quality check failed due to system error"
            )
```

### **Gemini API Integration for Reviews**

```python
class GeminiReviewAnalyzer:
    """Gemini-powered content analysis for review agents"""
    
    async def analyze_content_quality(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content using Gemini with specific review focus"""
        
        prompts = {
            "quality": self._build_quality_analysis_prompt(content),
            "brand": self._build_brand_analysis_prompt(content),
            "seo": self._build_seo_analysis_prompt(content)
        }
        
        try:
            response = await self.gemini_client.generate_content(
                prompt=prompts[analysis_type],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1500
            )
            
            # Extract structured analysis from response
            analysis_result = self._parse_gemini_analysis_response(response)
            
            # Track API usage for cost monitoring
            await self._track_gemini_usage(response.usage_metadata)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Gemini analysis failed for {analysis_type}: {e}")
            raise
    
    def _build_quality_analysis_prompt(self, content: str) -> str:
        """Build quality-focused analysis prompt"""
        return f"""
        Analyze this content for quality and accuracy:
        
        Content: {content}
        
        Provide analysis in JSON format:
        {{
            "accuracy_score": 0.0-1.0,
            "structure_score": 0.0-1.0,
            "clarity_score": 0.0-1.0,
            "relevance_score": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "issues": ["list", "of", "specific", "issues"],
            "recommendations": ["specific", "improvement", "suggestions"],
            "confidence": 0.0-1.0
        }}
        """
```

---

## **Integration with Existing Systems**

### **Campaign Orchestration Integration**

```python
# In existing content_generation_workflow.py
async def execute_content_generation_plan(self, campaign_id: str) -> Dict[str, Any]:
    """Enhanced content generation with review workflow integration"""
    
    # ... existing content generation logic ...
    
    # Launch review workflows for completed content
    review_results = []
    for task in plan.content_tasks:
        if task.status == ContentTaskStatus.COMPLETED and task.generated_content:
            
            # Initialize review workflow
            review_orchestrator = ReviewWorkflowOrchestrator()
            review_state = ReviewWorkflowState(
                content_id=task.generated_content.content_id,
                content_type=task.content_type.value,
                content_data=task.generated_content.to_dict(),
                campaign_id=campaign_id
            )
            
            # Launch review (will pause at human checkpoints)
            review_result = await review_orchestrator.execute_review_workflow(review_state)
            review_results.append(review_result)
            
            # Update task status based on review state
            if review_result.requires_human_review:
                task.status = ContentTaskStatus.REQUIRES_REVIEW
                task.review_workflow_id = review_result.workflow_id
    
    return {
        "generation_results": execution_results,
        "review_workflows": review_results
    }
```

### **API Endpoints for Review Management**

```python
# /src/api/routes/review_workflows.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

router = APIRouter(prefix="/api/v2/review-workflows", tags=["review-workflows"])

@router.get("/campaigns/{campaign_id}/reviews")
async def get_campaign_reviews(
    campaign_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None
) -> List[ReviewWorkflowResponse]:
    """Get all review workflows for a campaign"""
    
@router.get("/reviews/{workflow_id}")
async def get_review_workflow(workflow_id: str) -> ReviewWorkflowDetailResponse:
    """Get detailed information about a specific review workflow"""
    
@router.post("/reviews/{workflow_id}/feedback")
async def submit_review_feedback(
    workflow_id: str,
    feedback: ReviewFeedbackRequest
) -> ReviewFeedbackResponse:
    """Submit human reviewer feedback and resume workflow"""
    
@router.post("/reviews/{workflow_id}/resume")
async def resume_review_workflow(workflow_id: str) -> WorkflowResumeResponse:
    """Manually resume a paused review workflow"""
    
@router.get("/reviewers/{reviewer_id}/pending")
async def get_pending_reviews(
    reviewer_id: str,
    limit: int = 50
) -> List[PendingReviewResponse]:
    """Get pending reviews for a specific reviewer"""
```

---

## **Frontend Architecture**

### **Review Dashboard Component Structure**

```typescript
// /frontend/src/components/ReviewDashboard.tsx
interface ReviewDashboardProps {
  campaignId?: string;
  reviewerId?: string;
}

interface ReviewWorkflowState {
  workflowId: string;
  contentId: string;
  contentType: string;
  currentStage: ReviewStage;
  workflowStatus: ReviewStatus;
  progressPercentage: number;
  assignedReviewer?: string;
  deadline?: Date;
  automatedScores: Record<ReviewStage, number>;
  requiresAction: boolean;
}

const ReviewDashboard: React.FC<ReviewDashboardProps> = ({ campaignId, reviewerId }) => {
  const [workflows, setWorkflows] = useState<ReviewWorkflowState[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  
  // Real-time updates via WebSocket
  useEffect(() => {
    const wsConnection = new WebSocket(`ws://localhost:8000/ws/review-updates`);
    
    wsConnection.onmessage = (event) => {
      const update = JSON.parse(event.data);
      if (update.type === 'workflow_status_change') {
        updateWorkflowInState(update.workflowId, update.newStatus);
      }
    };
    
    return () => wsConnection.close();
  }, []);
  
  return (
    <div className="review-dashboard">
      <ReviewProgressBar workflows={workflows} />
      <ReviewWorkflowList 
        workflows={workflows}
        onSelectWorkflow={setSelectedWorkflow}
      />
      {selectedWorkflow && (
        <ReviewDetailModal 
          workflowId={selectedWorkflow}
          onClose={() => setSelectedWorkflow(null)}
        />
      )}
    </div>
  );
};
```

### **Stage-Specific Review Forms**

```typescript
// /frontend/src/components/ReviewStageForm.tsx
interface ReviewStageFormProps {
  workflowId: string;
  stage: ReviewStage;
  automatedAnalysis: AutomatedAnalysisResult;
  onSubmitFeedback: (feedback: ReviewFeedback) => void;
}

const ReviewStageForm: React.FC<ReviewStageFormProps> = ({
  workflowId,
  stage,
  automatedAnalysis,
  onSubmitFeedback
}) => {
  const [feedback, setFeedback] = useState<ReviewFeedback>({
    decision: ReviewDecision.PENDING,
    score: automatedAnalysis.confidence_score,
    comments: '',
    suggestions: [],
    flagged_issues: []
  });
  
  // Stage-specific form fields
  const renderStageSpecificFields = () => {
    switch (stage) {
      case ReviewStage.QUALITY_CHECK:
        return <QualityCheckForm feedback={feedback} onChange={setFeedback} />;
      case ReviewStage.BRAND_CHECK:
        return <BrandCheckForm feedback={feedback} onChange={setFeedback} />;
      case ReviewStage.SEO_REVIEW:
        return <SEOReviewForm feedback={feedback} onChange={setFeedback} />;
      case ReviewStage.FINAL_APPROVAL:
        return <FinalApprovalForm feedback={feedback} onChange={setFeedback} />;
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="review-stage-form">
      <AutomatedAnalysisDisplay analysis={automatedAnalysis} />
      {renderStageSpecificFields()}
      <ReviewDecisionButtons 
        onApprove={() => handleDecision(ReviewDecision.APPROVE)}
        onReject={() => handleDecision(ReviewDecision.REJECT)}
        onRequestChanges={() => handleDecision(ReviewDecision.REQUEST_CHANGES)}
      />
    </form>
  );
};
```

---

## **Performance Optimizations**

### **Async Processing Pattern**

```python
# Prevent review workflow from blocking content generation
async def launch_review_workflow_async(content_data: Dict[str, Any]) -> str:
    """Launch review workflow asynchronously"""
    
    # Create workflow task that runs independently
    workflow_task = asyncio.create_task(
        review_orchestrator.execute_review_workflow(review_state)
    )
    
    # Return workflow ID immediately, don't wait for completion
    workflow_id = f"review_{content_data['content_id']}"
    
    # Store task reference for monitoring
    active_review_workflows[workflow_id] = workflow_task
    
    return workflow_id
```

### **Intelligent Caching**

```python
class ReviewAnalysisCache:
    """Cache expensive AI analysis results"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    async def get_cached_analysis(
        self, 
        content_hash: str, 
        analysis_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        cache_key = f"review_analysis:{analysis_type}:{content_hash}"
        cached_result = await self.redis.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    async def cache_analysis_result(
        self,
        content_hash: str,
        analysis_type: str, 
        result: Dict[str, Any]
    ):
        """Cache analysis result with TTL"""
        cache_key = f"review_analysis:{analysis_type}:{content_hash}"
        await self.redis.setex(
            cache_key, 
            self.cache_ttl,
            json.dumps(result)
        )
```

### **Database Query Optimization**

```sql
-- Optimized query for reviewer dashboard
SELECT 
    crw.id as workflow_id,
    crw.content_id,
    crw.current_stage,
    crw.workflow_status,
    rc.deadline,
    rc.automated_score,
    COUNT(*) OVER (PARTITION BY rc.reviewer_id) as total_assigned
FROM content_review_workflows crw
JOIN review_checkpoints rc ON crw.workflow_execution_id = rc.workflow_execution_id
WHERE rc.reviewer_id = $1 
    AND rc.status = 'pending'
    AND crw.is_paused = true
ORDER BY rc.deadline ASC, rc.created_at ASC
LIMIT 50;

-- Index to support this query
CREATE INDEX idx_reviewer_pending_reviews ON review_checkpoints(reviewer_id, status, deadline) 
WHERE status = 'pending';
```

---

## **Monitoring and Observability**

### **Review Workflow Metrics**

```python
class ReviewWorkflowMetrics:
    """Comprehensive metrics for review workflow performance"""
    
    def __init__(self, metrics_client):
        self.metrics = metrics_client
    
    def track_workflow_stage_completion(
        self,
        workflow_id: str,
        stage: ReviewStage,
        duration_ms: int,
        automated_score: float,
        human_required: bool
    ):
        """Track stage completion metrics"""
        self.metrics.histogram(
            'review_workflow.stage_duration_ms',
            duration_ms,
            tags={
                'stage': stage.value,
                'automated': str(not human_required),
                'workflow_id': workflow_id
            }
        )
        
        self.metrics.gauge(
            'review_workflow.automated_confidence_score',
            automated_score,
            tags={'stage': stage.value}
        )
    
    def track_human_review_metrics(
        self,
        stage: ReviewStage,
        reviewer_id: str,
        decision: ReviewDecision,
        review_time_ms: int
    ):
        """Track human reviewer performance"""
        self.metrics.histogram(
            'review_workflow.human_review_time_ms',
            review_time_ms,
            tags={
                'stage': stage.value,
                'reviewer': reviewer_id,
                'decision': decision.value
            }
        )
        
        self.metrics.counter(
            'review_workflow.human_decisions',
            1,
            tags={
                'stage': stage.value,
                'decision': decision.value
            }
        )
```

---

## **Security Considerations**

### **Access Control**

```python
class ReviewWorkflowAccessControl:
    """Ensure reviewers only access assigned content"""
    
    async def validate_reviewer_access(
        self,
        reviewer_id: str,
        workflow_id: str,
        stage: ReviewStage
    ) -> bool:
        """Validate reviewer has access to specific workflow stage"""
        
        # Check if reviewer is assigned to this stage
        checkpoint = await self.db.fetch_checkpoint(workflow_id, stage)
        if not checkpoint or checkpoint.reviewer_id != reviewer_id:
            return False
        
        # Check if reviewer has role permissions for stage
        reviewer_permissions = await self.get_reviewer_permissions(reviewer_id)
        required_permission = f"review:{stage.value}"
        
        return required_permission in reviewer_permissions
    
    async def mask_sensitive_content(
        self,
        content: Dict[str, Any],
        reviewer_role: str
    ) -> Dict[str, Any]:
        """Mask sensitive information based on reviewer role"""
        
        # Implementation depends on your sensitivity requirements
        if reviewer_role not in ['senior_reviewer', 'compliance_officer']:
            # Remove or mask sensitive financial data
            content = self._remove_sensitive_fields(content)
        
        return content
```

This technical architecture provides a comprehensive, production-ready foundation for implementing the 4-stage Review Workflow system with full LangGraph integration, human-in-the-loop capabilities, and performance optimizations.