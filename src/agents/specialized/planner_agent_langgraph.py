"""
LangGraph-based Planner Agent with advanced strategic planning workflow.

This agent creates comprehensive content and campaign plans using sophisticated workflows
with strategic analysis, competitive assessment, and execution roadmaps.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.langgraph_base import LangGraphWorkflowBase, WorkflowState
from ..core.base_agent import AgentType, AgentResult, AgentMetadata

logger = logging.getLogger(__name__)

@dataclass
class PlannerState(WorkflowState):
    """State for Planner LangGraph workflow."""
    # Input requirements
    planning_objective: str = ""
    business_context: Dict[str, Any] = field(default_factory=dict)
    target_audience: str = "general"
    timeline: str = "1 month"
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    
    # Strategic analysis
    market_analysis: Dict[str, Any] = field(default_factory=dict)
    competitor_analysis: List[Dict[str, Any]] = field(default_factory=list)
    opportunity_assessment: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: List[Dict[str, Any]] = field(default_factory=list)
    
    # Planning components
    strategic_goals: List[Dict[str, Any]] = field(default_factory=list)
    tactical_initiatives: List[Dict[str, Any]] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    execution_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Deliverables
    content_calendar: List[Dict[str, Any]] = field(default_factory=list)
    campaign_roadmap: Dict[str, Any] = field(default_factory=dict)
    success_framework: Dict[str, Any] = field(default_factory=dict)
    
    # Quality control
    plan_completeness_score: float = 0.0
    feasibility_score: float = 0.0
    alignment_score: float = 0.0
    
    # Workflow control
    requires_refinement: bool = False
    refinement_areas: List[str] = field(default_factory=list)
    planning_iterations: int = 0
    max_iterations: int = 2

class PlannerAgentLangGraph(LangGraphWorkflowBase[PlannerState]):
    """
    LangGraph-based Planner with sophisticated strategic planning workflow.
    """
    
    def __init__(self, workflow_name: str = "Planner_workflow"):
        super().__init__(workflow_name=workflow_name)
        logger.info("PlannerAgentLangGraph initialized with advanced planning capabilities")
    
    def _create_workflow_graph(self):
        """Create the LangGraph workflow structure."""
        from src.agents.core.langgraph_compat import StateGraph
        
        workflow = StateGraph(PlannerState)
        
        # Define workflow nodes
        workflow.add_node("analyze_context", self._analyze_context)
        workflow.add_node("conduct_market_analysis", self._conduct_market_analysis)
        workflow.add_node("assess_competition", self._assess_competition)
        workflow.add_node("identify_opportunities", self._identify_opportunities)
        workflow.add_node("assess_risks", self._assess_risks)
        workflow.add_node("define_strategic_goals", self._define_strategic_goals)
        workflow.add_node("develop_tactical_plans", self._develop_tactical_plans)
        workflow.add_node("create_execution_roadmap", self._create_execution_roadmap)
        workflow.add_node("build_success_framework", self._build_success_framework)
        workflow.add_node("validate_plan", self._validate_plan)
        workflow.add_node("refine_plan", self._refine_plan)
        workflow.add_node("finalize_plan", self._finalize_plan)
        
        # Define workflow edges
        workflow.set_entry_point("analyze_context")
        
        workflow.add_edge("analyze_context", "conduct_market_analysis")
        workflow.add_edge("conduct_market_analysis", "assess_competition")
        workflow.add_edge("assess_competition", "identify_opportunities")
        workflow.add_edge("identify_opportunities", "assess_risks")
        workflow.add_edge("assess_risks", "define_strategic_goals")
        workflow.add_edge("define_strategic_goals", "develop_tactical_plans")
        workflow.add_edge("develop_tactical_plans", "create_execution_roadmap")
        workflow.add_edge("create_execution_roadmap", "build_success_framework")
        workflow.add_edge("build_success_framework", "validate_plan")
        
        # Conditional routing based on plan validation
        workflow.add_conditional_edges(
            "validate_plan",
            self._should_refine_plan,
            {
                "refine": "refine_plan",
                "finalize": "finalize_plan"
            }
        )
        
        workflow.add_edge("refine_plan", "define_strategic_goals")
        workflow.set_finish_point("finalize_plan")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def _create_initial_state(self, input_data: Dict[str, Any]) -> PlannerState:
        """Create initial workflow state from input."""
        return PlannerState(
            planning_objective=input_data.get("planning_objective", input_data.get("objective", "")),
            business_context=input_data.get("business_context", {}),
            target_audience=input_data.get("target_audience", "general"),
            timeline=input_data.get("timeline", "1 month"),
            budget_constraints=input_data.get("budget_constraints", {}),
            success_metrics=input_data.get("success_metrics", []),
            workflow_id=self.workflow_id,
            agent_name=self.metadata.name,
            current_step="analyze_context"
        )
    
    def _analyze_context(self, state: PlannerState) -> PlannerState:
        """Analyze business context and planning requirements."""
        logger.info("Analyzing business context and planning requirements")
        
        context_analysis = {
            "objective_type": self._classify_objective(state.planning_objective),
            "scope_assessment": self._assess_scope(state.planning_objective, state.timeline),
            "stakeholder_identification": self._identify_stakeholders(state.business_context),
            "constraint_analysis": self._analyze_constraints(state.budget_constraints, state.timeline),
            "success_criteria": state.success_metrics or self._define_default_metrics(state.planning_objective)
        }
        
        # Update business context with analysis
        state.business_context.update({
            "context_analysis": context_analysis,
            "planning_parameters": {
                "complexity_level": "high" if len(state.success_metrics) > 5 else "medium",
                "resource_intensity": "high" if state.budget_constraints else "medium",
                "timeline_pressure": "high" if "week" in state.timeline else "medium"
            }
        })
        
        state.current_step = "conduct_market_analysis"
        
        return state
    
    def _conduct_market_analysis(self, state: PlannerState) -> PlannerState:
        """Conduct market analysis for strategic planning."""
        logger.info("Conducting market analysis")
        
        # Simulate comprehensive market analysis
        market_analysis = {
            "market_size": {
                "total_addressable_market": "Large",
                "serviceable_addressable_market": "Medium",
                "growth_rate": "15% annually"
            },
            "market_trends": [
                "Increasing digital transformation adoption",
                "Growing demand for automated solutions", 
                "Shift towards subscription-based models",
                "Rising importance of data privacy"
            ],
            "customer_segments": [
                {
                    "segment": "Enterprise clients",
                    "size": "40%",
                    "characteristics": ["High budget", "Complex needs", "Long sales cycles"],
                    "pain_points": ["Integration challenges", "ROI measurement", "Change management"]
                },
                {
                    "segment": "SMB clients", 
                    "size": "35%",
                    "characteristics": ["Budget conscious", "Quick decisions", "Simple needs"],
                    "pain_points": ["Cost concerns", "Limited resources", "Technical expertise"]
                },
                {
                    "segment": "Startups",
                    "size": "25%", 
                    "characteristics": ["Innovation focused", "Rapid growth", "Resource limited"],
                    "pain_points": ["Scalability", "Funding constraints", "Market validation"]
                }
            ],
            "market_dynamics": {
                "competitive_intensity": "High",
                "barriers_to_entry": "Medium",
                "supplier_power": "Low",
                "buyer_power": "Medium"
            }
        }
        
        state.market_analysis = market_analysis
        state.current_step = "assess_competition"
        
        return state
    
    def _assess_competition(self, state: PlannerState) -> PlannerState:
        """Assess competitive landscape."""
        logger.info("Assessing competitive landscape")
        
        # Simulate competitive analysis
        competitor_analysis = [
            {
                "competitor": "Market Leader A",
                "market_share": "25%",
                "strengths": ["Brand recognition", "Resource advantage", "Market presence"],
                "weaknesses": ["Innovation lag", "High prices", "Complex solutions"],
                "strategic_focus": ["Market expansion", "Premium positioning"],
                "threat_level": "High"
            },
            {
                "competitor": "Challenger B",
                "market_share": "15%", 
                "strengths": ["Innovation", "Agility", "Customer focus"],
                "weaknesses": ["Limited resources", "Small market presence"],
                "strategic_focus": ["Product innovation", "Niche markets"],
                "threat_level": "Medium"
            },
            {
                "competitor": "Disruptor C",
                "market_share": "8%",
                "strengths": ["Technology advantage", "Low cost", "Fast growth"],
                "weaknesses": ["Limited features", "Unproven model"],
                "strategic_focus": ["Mass market", "Technology disruption"], 
                "threat_level": "Medium"
            }
        ]
        
        state.competitor_analysis = competitor_analysis
        state.current_step = "identify_opportunities"
        
        return state
    
    def _identify_opportunities(self, state: PlannerState) -> PlannerState:
        """Identify strategic opportunities."""
        logger.info("Identifying strategic opportunities")
        
        # Analyze opportunities based on market and competitive analysis
        opportunities = {
            "market_opportunities": [
                {
                    "opportunity": "Underserved SMB segment",
                    "potential": "High",
                    "effort_required": "Medium",
                    "timeframe": "3-6 months",
                    "success_probability": "80%"
                },
                {
                    "opportunity": "Emerging technology integration",
                    "potential": "High", 
                    "effort_required": "High",
                    "timeframe": "6-12 months",
                    "success_probability": "60%"
                },
                {
                    "opportunity": "Geographic expansion",
                    "potential": "Medium",
                    "effort_required": "High", 
                    "timeframe": "9-18 months",
                    "success_probability": "70%"
                }
            ],
            "competitive_opportunities": [
                {
                    "opportunity": "Price-value positioning gap",
                    "competitive_advantage": "Better value proposition",
                    "market_impact": "Medium",
                    "implementation_complexity": "Low"
                },
                {
                    "opportunity": "Feature differentiation",
                    "competitive_advantage": "Unique capabilities",
                    "market_impact": "High", 
                    "implementation_complexity": "Medium"
                }
            ],
            "strategic_priorities": [
                "Market share growth in target segments",
                "Product differentiation and innovation",
                "Customer experience optimization",
                "Operational efficiency improvement"
            ]
        }
        
        state.opportunity_assessment = opportunities
        state.current_step = "assess_risks"
        
        return state
    
    def _assess_risks(self, state: PlannerState) -> PlannerState:
        """Assess strategic and operational risks."""
        logger.info("Assessing strategic and operational risks")
        
        risk_analysis = [
            {
                "risk_category": "Market Risk",
                "specific_risks": [
                    {
                        "risk": "Market saturation",
                        "probability": "Medium",
                        "impact": "High",
                        "mitigation": "Diversify into adjacent markets"
                    },
                    {
                        "risk": "Economic downturn",
                        "probability": "Low",
                        "impact": "High", 
                        "mitigation": "Build financial reserves and flexible cost structure"
                    }
                ]
            },
            {
                "risk_category": "Competitive Risk",
                "specific_risks": [
                    {
                        "risk": "New market entrant",
                        "probability": "High",
                        "impact": "Medium",
                        "mitigation": "Strengthen competitive moats and customer loyalty"
                    },
                    {
                        "risk": "Price competition",
                        "probability": "Medium",
                        "impact": "Medium",
                        "mitigation": "Focus on value differentiation rather than price"
                    }
                ]
            },
            {
                "risk_category": "Operational Risk",
                "specific_risks": [
                    {
                        "risk": "Resource constraints",
                        "probability": "Medium", 
                        "impact": "Medium",
                        "mitigation": "Phased implementation and resource optimization"
                    },
                    {
                        "risk": "Execution delays",
                        "probability": "Medium",
                        "impact": "Medium",
                        "mitigation": "Clear milestones and contingency planning"
                    }
                ]
            }
        ]
        
        state.risk_analysis = risk_analysis
        state.current_step = "define_strategic_goals"
        
        return state
    
    def _define_strategic_goals(self, state: PlannerState) -> PlannerState:
        """Define strategic goals based on analysis."""
        logger.info("Defining strategic goals")
        
        # Create SMART goals based on opportunities and constraints
        strategic_goals = [
            {
                "goal": "Increase market share in target segment by 15%",
                "category": "Growth",
                "timeline": self._parse_timeline_months(state.timeline),
                "success_metrics": ["Market share percentage", "Revenue growth", "Customer acquisition"],
                "priority": "High",
                "resource_requirements": ["Marketing budget", "Sales team expansion"],
                "dependencies": ["Market research", "Competitive positioning"]
            },
            {
                "goal": "Launch 2 new product features based on customer feedback",
                "category": "Innovation",
                "timeline": int(self._parse_timeline_months(state.timeline) * 0.8),  # 80% of timeline
                "success_metrics": ["Feature adoption rate", "Customer satisfaction", "Revenue impact"],
                "priority": "High",
                "resource_requirements": ["Development team", "R&D budget"],
                "dependencies": ["Customer research", "Technical feasibility"]
            },
            {
                "goal": "Improve customer satisfaction score by 20%",
                "category": "Customer Experience",
                "timeline": self._parse_timeline_months(state.timeline),
                "success_metrics": ["NPS score", "Customer retention rate", "Support ticket resolution time"],
                "priority": "Medium", 
                "resource_requirements": ["Customer success team", "Process improvements"],
                "dependencies": ["Customer feedback analysis", "Service optimization"]
            }
        ]
        
        state.strategic_goals = strategic_goals
        state.current_step = "develop_tactical_plans"
        
        return state
    
    def _develop_tactical_plans(self, state: PlannerState) -> PlannerState:
        """Develop tactical initiatives to achieve strategic goals."""
        logger.info("Developing tactical initiatives")
        
        tactical_initiatives = []
        
        for goal in state.strategic_goals:
            # Create tactical initiatives for each strategic goal
            if "market share" in goal["goal"].lower():
                initiatives = [
                    {
                        "initiative": "Launch targeted marketing campaign",
                        "goal_alignment": goal["goal"],
                        "description": "Multi-channel marketing campaign focusing on target segments",
                        "timeline": "1-3 months",
                        "budget_allocation": "40%",
                        "success_metrics": ["Lead generation", "Campaign ROI", "Brand awareness"],
                        "action_items": [
                            "Develop campaign messaging and creative",
                            "Select marketing channels and budget allocation",
                            "Launch campaign and monitor performance",
                            "Optimize based on performance data"
                        ]
                    },
                    {
                        "initiative": "Enhance sales process efficiency",
                        "goal_alignment": goal["goal"],
                        "description": "Streamline sales process and improve conversion rates",
                        "timeline": "2-4 months",
                        "budget_allocation": "20%",
                        "success_metrics": ["Sales cycle length", "Conversion rate", "Sales productivity"],
                        "action_items": [
                            "Analyze current sales process",
                            "Identify bottlenecks and improvement opportunities",
                            "Implement process improvements and training",
                            "Monitor and refine based on results"
                        ]
                    }
                ]
                tactical_initiatives.extend(initiatives)
            
            elif "product features" in goal["goal"].lower():
                initiatives = [
                    {
                        "initiative": "Customer-driven feature development",
                        "goal_alignment": goal["goal"],
                        "description": "Develop features based on customer feedback and market needs",
                        "timeline": "2-6 months",
                        "budget_allocation": "60%",
                        "success_metrics": ["Feature completion", "User adoption", "Customer feedback scores"],
                        "action_items": [
                            "Conduct customer interviews and surveys",
                            "Prioritize features based on impact and feasibility",
                            "Develop and test features",
                            "Launch and monitor adoption"
                        ]
                    }
                ]
                tactical_initiatives.extend(initiatives)
            
            elif "customer satisfaction" in goal["goal"].lower():
                initiatives = [
                    {
                        "initiative": "Customer experience optimization",
                        "goal_alignment": goal["goal"],
                        "description": "Improve customer touchpoints and support processes",
                        "timeline": "1-4 months",
                        "budget_allocation": "30%", 
                        "success_metrics": ["Customer satisfaction scores", "Support response time", "Issue resolution rate"],
                        "action_items": [
                            "Map customer journey and identify pain points",
                            "Implement process improvements",
                            "Train customer-facing teams",
                            "Measure and iterate based on feedback"
                        ]
                    }
                ]
                tactical_initiatives.extend(initiatives)
        
        state.tactical_initiatives = tactical_initiatives
        state.current_step = "create_execution_roadmap"
        
        return state
    
    def _create_execution_roadmap(self, state: PlannerState) -> PlannerState:
        """Create detailed execution roadmap."""
        logger.info("Creating execution roadmap")
        
        timeline_months = self._parse_timeline_months(state.timeline)
        
        # Create execution phases
        execution_phases = []
        
        # Phase 1: Foundation (first 25% of timeline)
        phase1_duration = max(1, int(timeline_months * 0.25))
        phase1 = {
            "phase": "Foundation & Setup",
            "duration_months": phase1_duration,
            "start_month": 1,
            "end_month": phase1_duration,
            "objectives": [
                "Establish project governance and team structure",
                "Complete detailed planning and resource allocation",
                "Begin foundational initiatives"
            ],
            "key_activities": [
                "Team formation and role assignment",
                "Detailed project planning and scheduling",
                "Baseline measurement and KPI setup",
                "Initial market research and customer interviews"
            ],
            "deliverables": [
                "Project charter and governance structure",
                "Detailed project plans and timelines", 
                "Baseline metrics and measurement framework",
                "Market research findings and insights"
            ],
            "success_criteria": [
                "Team and governance in place",
                "Plans approved and resources allocated",
                "Baseline metrics established"
            ]
        }
        execution_phases.append(phase1)
        
        # Phase 2: Implementation (middle 50% of timeline)
        phase2_start = phase1_duration + 1
        phase2_duration = max(1, int(timeline_months * 0.5))
        phase2_end = phase2_start + phase2_duration - 1
        phase2 = {
            "phase": "Core Implementation",
            "duration_months": phase2_duration,
            "start_month": phase2_start,
            "end_month": phase2_end,
            "objectives": [
                "Execute core tactical initiatives",
                "Launch key products/services/campaigns",
                "Monitor progress and make adjustments"
            ],
            "key_activities": [
                "Execute marketing campaigns and sales initiatives",
                "Develop and launch new features/products",
                "Implement customer experience improvements",
                "Regular performance monitoring and optimization"
            ],
            "deliverables": [
                "Marketing campaigns launched and running",
                "New features/products in market",
                "Improved customer processes and touchpoints",
                "Regular performance reports and insights"
            ],
            "success_criteria": [
                "All major initiatives launched successfully",
                "Performance metrics showing positive trends",
                "Customer feedback indicating improvements"
            ]
        }
        execution_phases.append(phase2)
        
        # Phase 3: Optimization (final 25% of timeline)
        phase3_start = phase2_end + 1
        phase3_duration = timeline_months - phase2_end
        phase3 = {
            "phase": "Optimization & Scale",
            "duration_months": phase3_duration,
            "start_month": phase3_start,
            "end_month": timeline_months,
            "objectives": [
                "Optimize performance based on results",
                "Scale successful initiatives",
                "Prepare for next planning cycle"
            ],
            "key_activities": [
                "Performance analysis and optimization",
                "Scaling successful programs",
                "Knowledge capture and documentation",
                "Next cycle planning preparation"
            ],
            "deliverables": [
                "Performance optimization recommendations",
                "Scaled programs and processes",
                "Lessons learned and best practices documentation",
                "Next planning cycle inputs"
            ],
            "success_criteria": [
                "Strategic goals achieved or on track",
                "Optimized processes and programs in place",
                "Clear path forward for continued success"
            ]
        }
        execution_phases.append(phase3)
        
        # Create content calendar (if applicable)
        content_calendar = []
        if "marketing" in state.planning_objective.lower() or "content" in state.planning_objective.lower():
            for month in range(1, timeline_months + 1):
                content_calendar.append({
                    "month": month,
                    "content_themes": [
                        f"Theme {month}.1: Industry insights and trends",
                        f"Theme {month}.2: Product features and benefits", 
                        f"Theme {month}.3: Customer success stories"
                    ],
                    "content_types": [
                        "Blog posts (4-6 per month)",
                        "Social media content (daily)",
                        "Email newsletters (weekly)",
                        "Case studies (1-2 per month)"
                    ],
                    "key_messages": [
                        "Position as industry thought leader",
                        "Highlight unique value proposition",
                        "Build trust through customer validation"
                    ]
                })
        
        # Create campaign roadmap
        campaign_roadmap = {
            "strategic_campaigns": [
                {
                    "campaign": "Brand Awareness Campaign",
                    "timeline": f"Months 1-{int(timeline_months/2)}",
                    "objectives": ["Increase brand recognition", "Generate qualified leads"],
                    "channels": ["Digital advertising", "Content marketing", "PR"],
                    "budget_allocation": "40%"
                },
                {
                    "campaign": "Product Launch Campaign", 
                    "timeline": f"Months {int(timeline_months/3)}-{int(timeline_months*2/3)}",
                    "objectives": ["Launch new features", "Drive product adoption"],
                    "channels": ["Product marketing", "Customer communications", "Sales enablement"],
                    "budget_allocation": "35%"
                },
                {
                    "campaign": "Customer Retention Campaign",
                    "timeline": f"Months {int(timeline_months/2)}-{timeline_months}",
                    "objectives": ["Improve customer satisfaction", "Reduce churn"],
                    "channels": ["Customer success", "Support optimization", "Loyalty programs"],
                    "budget_allocation": "25%"
                }
            ]
        }
        
        state.execution_phases = execution_phases
        state.content_calendar = content_calendar
        state.campaign_roadmap = campaign_roadmap
        state.current_step = "build_success_framework"
        
        return state
    
    def _build_success_framework(self, state: PlannerState) -> PlannerState:
        """Build comprehensive success measurement framework."""
        logger.info("Building success measurement framework")
        
        success_framework = {
            "key_performance_indicators": [
                {
                    "category": "Growth Metrics",
                    "kpis": [
                        {"metric": "Revenue Growth", "target": "15% increase", "measurement_frequency": "Monthly"},
                        {"metric": "Market Share", "target": "15% increase", "measurement_frequency": "Quarterly"},
                        {"metric": "Customer Acquisition", "target": "20% increase", "measurement_frequency": "Monthly"}
                    ]
                },
                {
                    "category": "Customer Metrics", 
                    "kpis": [
                        {"metric": "Customer Satisfaction (NPS)", "target": "20% improvement", "measurement_frequency": "Quarterly"},
                        {"metric": "Customer Retention Rate", "target": "5% improvement", "measurement_frequency": "Monthly"},
                        {"metric": "Customer Lifetime Value", "target": "10% increase", "measurement_frequency": "Quarterly"}
                    ]
                },
                {
                    "category": "Operational Metrics",
                    "kpis": [
                        {"metric": "Sales Cycle Length", "target": "15% reduction", "measurement_frequency": "Monthly"},
                        {"metric": "Lead Conversion Rate", "target": "10% improvement", "measurement_frequency": "Monthly"},
                        {"metric": "Support Response Time", "target": "25% reduction", "measurement_frequency": "Weekly"}
                    ]
                }
            ],
            "measurement_framework": {
                "data_sources": ["CRM system", "Analytics platform", "Customer surveys", "Financial reports"],
                "reporting_schedule": {
                    "weekly": ["Operational metrics", "Campaign performance"],
                    "monthly": ["Growth metrics", "Customer metrics", "Financial performance"],
                    "quarterly": ["Strategic goal progress", "Market position", "Competitive analysis"]
                },
                "review_processes": {
                    "weekly_reviews": "Team performance check-ins",
                    "monthly_reviews": "Department performance and goal progress",
                    "quarterly_reviews": "Strategic plan assessment and adjustments"
                }
            },
            "success_milestones": [
                {
                    "milestone": f"Month {int(self._parse_timeline_months(state.timeline) * 0.25)} - Foundation Complete",
                    "criteria": ["Team and processes in place", "Baseline metrics established", "Initial campaigns launched"]
                },
                {
                    "milestone": f"Month {int(self._parse_timeline_months(state.timeline) * 0.5)} - Mid-point Assessment",
                    "criteria": ["50% progress toward goals", "Key initiatives showing results", "Performance on track"]
                },
                {
                    "milestone": f"Month {int(self._parse_timeline_months(state.timeline) * 0.75)} - Final Push",
                    "criteria": ["75% progress toward goals", "Optimization initiatives implemented", "Clear path to success"]
                },
                {
                    "milestone": f"Month {self._parse_timeline_months(state.timeline)} - Goal Achievement",
                    "criteria": ["Strategic goals achieved", "Success metrics met", "Next phase ready"]
                }
            ]
        }
        
        state.success_framework = success_framework
        state.current_step = "validate_plan"
        
        return state
    
    def _validate_plan(self, state: PlannerState) -> PlannerState:
        """Validate the comprehensive plan."""
        logger.info("Validating comprehensive plan")
        
        # Assess plan completeness
        completeness_factors = []
        
        # Strategic components
        has_goals = len(state.strategic_goals) > 0
        has_tactics = len(state.tactical_initiatives) > 0
        has_roadmap = len(state.execution_phases) > 0
        has_metrics = len(state.success_framework.get("key_performance_indicators", [])) > 0
        
        completeness_factors.extend([has_goals, has_tactics, has_roadmap, has_metrics])
        
        # Analysis components
        has_market_analysis = bool(state.market_analysis)
        has_competitor_analysis = len(state.competitor_analysis) > 0
        has_opportunity_assessment = bool(state.opportunity_assessment)
        has_risk_analysis = len(state.risk_analysis) > 0
        
        completeness_factors.extend([has_market_analysis, has_competitor_analysis, 
                                   has_opportunity_assessment, has_risk_analysis])
        
        plan_completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Assess feasibility
        feasibility_factors = []
        
        # Resource feasibility
        resource_requirements = sum(len(goal.get("resource_requirements", [])) for goal in state.strategic_goals)
        resource_feasibility = 1.0 if resource_requirements <= 10 else 0.8  # Simple heuristic
        feasibility_factors.append(resource_feasibility)
        
        # Timeline feasibility  
        timeline_months = self._parse_timeline_months(state.timeline)
        initiative_count = len(state.tactical_initiatives)
        timeline_feasibility = 1.0 if initiative_count / timeline_months <= 3 else 0.7  # Max 3 initiatives per month
        feasibility_factors.append(timeline_feasibility)
        
        # Budget feasibility (if constraints exist)
        if state.budget_constraints:
            budget_feasibility = 0.8  # Assume some budget pressure
        else:
            budget_feasibility = 1.0
        feasibility_factors.append(budget_feasibility)
        
        feasibility_score = sum(feasibility_factors) / len(feasibility_factors)
        
        # Assess alignment
        alignment_factors = []
        
        # Goal-objective alignment
        objective_keywords = set(state.planning_objective.lower().split())
        goal_alignment_scores = []
        for goal in state.strategic_goals:
            goal_keywords = set(goal["goal"].lower().split())
            alignment = len(objective_keywords.intersection(goal_keywords)) / len(objective_keywords.union(goal_keywords))
            goal_alignment_scores.append(alignment)
        
        avg_goal_alignment = sum(goal_alignment_scores) / len(goal_alignment_scores) if goal_alignment_scores else 0.5
        alignment_factors.append(avg_goal_alignment)
        
        # Tactics-goals alignment
        tactics_with_goals = sum(1 for tactic in state.tactical_initiatives if tactic.get("goal_alignment"))
        tactics_alignment = tactics_with_goals / len(state.tactical_initiatives) if state.tactical_initiatives else 0
        alignment_factors.append(tactics_alignment)
        
        alignment_score = sum(alignment_factors) / len(alignment_factors)
        
        # Update state with validation scores
        state.plan_completeness_score = plan_completeness_score
        state.feasibility_score = feasibility_score
        state.alignment_score = alignment_score
        
        # Identify refinement areas
        refinement_areas = []
        if plan_completeness_score < 0.8:
            refinement_areas.append("Plan completeness - missing key components")
        if feasibility_score < 0.7:
            refinement_areas.append("Plan feasibility - resource or timeline constraints")
        if alignment_score < 0.7:
            refinement_areas.append("Strategic alignment - goals and tactics not well aligned")
        
        state.refinement_areas = refinement_areas
        state.current_step = "finalize_plan"
        
        return state
    
    def _should_refine_plan(self, state: PlannerState) -> str:
        """Determine if plan needs refinement."""
        avg_score = (state.plan_completeness_score + state.feasibility_score + state.alignment_score) / 3
        
        needs_refinement = (
            avg_score < 0.75 or
            len(state.refinement_areas) > 1
        ) and state.planning_iterations < state.max_iterations
        
        if needs_refinement:
            state.requires_refinement = True
            state.planning_iterations += 1
            return "refine"
        else:
            return "finalize"
    
    def _refine_plan(self, state: PlannerState) -> PlannerState:
        """Refine plan based on validation feedback."""
        logger.info(f"Refining plan (iteration {state.planning_iterations})")
        
        # Address specific refinement areas
        for area in state.refinement_areas:
            if "completeness" in area.lower():
                # Add missing strategic goal if needed
                if len(state.strategic_goals) < 3:
                    additional_goal = {
                        "goal": "Improve operational efficiency by 10%",
                        "category": "Operations",
                        "timeline": self._parse_timeline_months(state.timeline),
                        "success_metrics": ["Process efficiency", "Cost reduction", "Team productivity"],
                        "priority": "Medium",
                        "resource_requirements": ["Process improvement team", "Technology tools"],
                        "dependencies": ["Current state assessment", "Change management"]
                    }
                    state.strategic_goals.append(additional_goal)
            
            elif "feasibility" in area.lower():
                # Adjust timelines or resource requirements
                for goal in state.strategic_goals:
                    if goal["timeline"] < 2:  # Very short timeline
                        goal["timeline"] = min(goal["timeline"] + 1, self._parse_timeline_months(state.timeline))
            
            elif "alignment" in area.lower():
                # Improve goal-objective alignment
                for goal in state.strategic_goals:
                    if state.planning_objective.lower() in ["growth", "expansion"]:
                        if "growth" not in goal["goal"].lower():
                            goal["goal"] = goal["goal"].replace("Improve", "Grow").replace("Increase", "Accelerate")
        
        state.current_step = "validate_plan"
        
        return state
    
    def _finalize_plan(self, state: PlannerState) -> PlannerState:
        """Finalize the comprehensive strategic plan."""
        logger.info("Finalizing comprehensive strategic plan")
        
        # Add final metadata and summary
        state.metadata.update({
            "strategic_goals_count": len(state.strategic_goals),
            "tactical_initiatives_count": len(state.tactical_initiatives),
            "execution_phases_count": len(state.execution_phases),
            "planning_timeline_months": self._parse_timeline_months(state.timeline),
            "plan_completeness_score": state.plan_completeness_score,
            "plan_feasibility_score": state.feasibility_score,
            "strategic_alignment_score": state.alignment_score,
            "overall_plan_quality": (state.plan_completeness_score + state.feasibility_score + state.alignment_score) / 3,
            "planning_iterations": state.planning_iterations,
            "content_calendar_months": len(state.content_calendar),
            "strategic_campaigns_count": len(state.campaign_roadmap.get("strategic_campaigns", [])),
            "planning_complete": True
        })
        
        state.current_step = "completed"
        
        return state
    
    # Helper methods
    def _classify_objective(self, objective: str) -> str:
        """Classify the planning objective type."""
        objective_lower = objective.lower()
        if any(word in objective_lower for word in ["growth", "expand", "increase"]):
            return "growth"
        elif any(word in objective_lower for word in ["launch", "new", "introduce"]):
            return "innovation"
        elif any(word in objective_lower for word in ["improve", "optimize", "enhance"]):
            return "optimization"
        elif any(word in objective_lower for word in ["market", "compete", "share"]):
            return "competitive"
        else:
            return "strategic"
    
    def _assess_scope(self, objective: str, timeline: str) -> str:
        """Assess the scope of the planning objective."""
        timeline_months = self._parse_timeline_months(timeline)
        objective_complexity = len(objective.split())
        
        if timeline_months <= 3 and objective_complexity <= 10:
            return "narrow"
        elif timeline_months <= 6 and objective_complexity <= 20:
            return "moderate"
        else:
            return "broad"
    
    def _identify_stakeholders(self, business_context: Dict[str, Any]) -> List[str]:
        """Identify key stakeholders based on business context."""
        stakeholders = ["Executive leadership", "Marketing team", "Sales team", "Product team"]
        
        # Add context-specific stakeholders
        if business_context.get("has_customers"):
            stakeholders.append("Key customers")
        if business_context.get("has_partners"):
            stakeholders.append("Strategic partners")
        if business_context.get("investor_backed"):
            stakeholders.append("Investors/Board")
            
        return stakeholders
    
    def _analyze_constraints(self, budget_constraints: Dict[str, Any], timeline: str) -> Dict[str, Any]:
        """Analyze planning constraints."""
        constraints = {
            "budget_limited": bool(budget_constraints),
            "timeline_pressure": "week" in timeline.lower() or "urgent" in timeline.lower(),
            "resource_constraints": budget_constraints.get("limited_resources", False),
            "regulatory_constraints": budget_constraints.get("compliance_requirements", False)
        }
        
        constraint_level = sum(constraints.values())
        constraints["overall_constraint_level"] = "high" if constraint_level >= 3 else "medium" if constraint_level >= 2 else "low"
        
        return constraints
    
    def _define_default_metrics(self, objective: str) -> List[str]:
        """Define default success metrics based on objective."""
        default_metrics = ["Revenue growth", "Customer satisfaction", "Market share"]
        
        objective_lower = objective.lower()
        if "marketing" in objective_lower:
            default_metrics.extend(["Lead generation", "Brand awareness", "Campaign ROI"])
        elif "product" in objective_lower:
            default_metrics.extend(["Product adoption", "User engagement", "Feature usage"])
        elif "sales" in objective_lower:
            default_metrics.extend(["Sales conversion", "Deal size", "Sales cycle time"])
        
        return default_metrics
    
    def _parse_timeline_months(self, timeline: str) -> int:
        """Parse timeline string to number of months."""
        timeline_lower = timeline.lower()
        
        if "week" in timeline_lower:
            weeks = 1
            try:
                weeks = int([word for word in timeline_lower.split() if word.isdigit()][0])
            except (IndexError, ValueError):
                pass
            return max(1, int(weeks / 4))  # Convert weeks to months
        elif "month" in timeline_lower:
            try:
                months = int([word for word in timeline_lower.split() if word.isdigit()][0])
                return max(1, months)
            except (IndexError, ValueError):
                return 1
        elif "quarter" in timeline_lower:
            try:
                quarters = int([word for word in timeline_lower.split() if word.isdigit()][0])
                return max(3, quarters * 3)
            except (IndexError, ValueError):
                return 3
        elif "year" in timeline_lower:
            try:
                years = int([word for word in timeline_lower.split() if word.isdigit()][0])
                return max(12, years * 12)
            except (IndexError, ValueError):
                return 12
        else:
            return 3  # Default to 3 months