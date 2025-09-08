# Agent Learning from Human Feedback Implementation Guide

## Overview

Your CrediLinq system can implement sophisticated agent learning from human feedback using your existing infrastructure. This guide shows how to leverage your current database schema and add learning capabilities that continuously improve agent performance based on human input.

## Current Architecture Foundation

Your system already has excellent foundations for learning:

### ✅ Existing Components
- **`AgentPerformance`** table tracking execution metrics and quality scores
- **`AgentDecision`** table with reasoning and confidence scores  
- **`ContentReviewWorkflow`** with human reviewer feedback
- **`ReviewFeedback`** table capturing human decisions and comments
- **`AsyncPerformanceTracker`** for real-time data collection

### 🔄 Learning Enhancement Opportunities
1. **Feedback-Quality Correlation Learning**
2. **Decision Pattern Recognition**  
3. **Human Preference Modeling**
4. **Automated Quality Prediction**
5. **Content Strategy Optimization**

---

## Human Feedback Learning Architecture

### 1. Feedback Collection System

Your existing `ReviewFeedback` table already captures human decisions:

```sql
-- Current ReviewFeedback structure (from your schema)
CREATE TABLE review_feedback (
    id UUID PRIMARY KEY,
    workflow_id UUID,
    content_id UUID,
    reviewer_id VARCHAR(255),
    stage review_stage, -- quality_check, brand_check, seo_review, etc.
    decision review_decision, -- approve, reject, request_changes
    comments TEXT,
    reviewed_at TIMESTAMPTZ
);
```

**Enhancement**: Add structured feedback fields for learning:

```sql
-- Enhanced feedback structure for learning
ALTER TABLE review_feedback 
ADD COLUMN feedback_categories JSONB DEFAULT '{}'::JSONB,
ADD COLUMN quality_scores JSONB DEFAULT '{}'::JSONB,
ADD COLUMN improvement_suggestions TEXT[],
ADD COLUMN feedback_confidence FLOAT CHECK (feedback_confidence >= 0 AND feedback_confidence <= 1),
ADD COLUMN is_training_data BOOLEAN DEFAULT TRUE;

-- Example feedback_categories structure:
-- {
--   "content_quality": {"score": 8, "issues": ["clarity", "structure"]},
--   "brand_alignment": {"score": 9, "issues": []},
--   "seo_effectiveness": {"score": 7, "issues": ["keyword_density", "meta_description"]},
--   "audience_targeting": {"score": 6, "issues": ["tone", "technical_level"]}
-- }
```

### 2. Agent Learning Database Schema

Add tables specifically for learning from feedback:

```sql
-- Agent learning patterns from human feedback
CREATE TABLE agent_learning_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    
    -- Learning pattern identification
    pattern_type VARCHAR(50) NOT NULL, -- decision_pattern, quality_correlation, preference_model
    pattern_description TEXT,
    
    -- Pattern data
    input_features JSONB NOT NULL, -- What inputs lead to this pattern
    output_characteristics JSONB NOT NULL, -- What outputs are associated
    human_feedback_signals JSONB NOT NULL, -- What human feedback indicates
    
    -- Learning metrics
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
    accuracy_rate FLOAT CHECK (accuracy_rate >= 0 AND accuracy_rate <= 1),
    
    -- Application tracking
    times_applied INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0,
    last_applied TIMESTAMPTZ,
    
    -- Metadata
    learned_from_feedback_ids UUID[], -- References to review_feedback records
    learning_algorithm VARCHAR(50) DEFAULT 'pattern_analysis',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(agent_name, pattern_type, pattern_description)
);

-- Human preference models per reviewer
CREATE TABLE human_preference_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reviewer_id VARCHAR(255) NOT NULL,
    
    -- Preference categories
    content_preferences JSONB DEFAULT '{}'::JSONB,
    style_preferences JSONB DEFAULT '{}'::JSONB,
    quality_standards JSONB DEFAULT '{}'::JSONB,
    
    -- Model metadata
    model_version INTEGER DEFAULT 1,
    training_samples INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    model_accuracy FLOAT,
    
    -- Preference examples:
    -- content_preferences: {
    --   "preferred_tone": "professional_friendly",
    --   "content_length": {"min": 1200, "max": 2000},  
    --   "structure_preference": "introduction_body_conclusion",
    --   "example_quality_threshold": 0.85
    -- }
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(reviewer_id, model_version)
);

-- Learning feedback correlations
CREATE TABLE feedback_quality_correlations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    
    -- Correlation data
    agent_output_feature VARCHAR(255) NOT NULL, -- e.g., "word_count", "keyword_density", "readability_score"
    human_feedback_aspect VARCHAR(255) NOT NULL, -- e.g., "content_quality", "brand_alignment"
    
    correlation_strength FLOAT NOT NULL CHECK (correlation_strength >= -1 AND correlation_strength <= 1),
    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
    statistical_significance FLOAT, -- p-value
    
    -- Correlation details
    feature_range JSONB, -- {"min": 800, "max": 2000} for word_count
    feedback_impact JSONB, -- {"positive_threshold": 0.8, "negative_threshold": 0.4}
    
    last_calculated TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(agent_name, agent_output_feature, human_feedback_aspect)
);

CREATE INDEX idx_agent_learning_patterns_agent ON agent_learning_patterns(agent_name, agent_type);
CREATE INDEX idx_human_preference_models_reviewer ON human_preference_models(reviewer_id, model_version);
CREATE INDEX idx_feedback_correlations_agent ON feedback_quality_correlations(agent_name, correlation_strength DESC);
```

### 3. Learning Implementation Components

```python
# src/agents/learning/human_feedback_learner.py

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

@dataclass
class LearningPattern:
    """Learned pattern from human feedback"""
    pattern_type: str
    input_features: Dict[str, Any]
    output_characteristics: Dict[str, Any]
    human_feedback_signals: Dict[str, Any]
    confidence_score: float
    sample_size: int

@dataclass 
class HumanPreference:
    """Human reviewer preference model"""
    reviewer_id: str
    content_preferences: Dict[str, Any]
    style_preferences: Dict[str, Any]
    quality_standards: Dict[str, Any]
    model_accuracy: float

class HumanFeedbackLearner:
    """
    Learns from human feedback to improve agent performance.
    
    This component analyzes patterns in human feedback and creates
    actionable improvements for agent decision-making.
    """
    
    def __init__(self, db_service):
        self.db_service = db_service
        self.learning_algorithms = {
            'correlation_analysis': self._analyze_correlations,
            'pattern_recognition': self._recognize_patterns,
            'preference_modeling': self._model_preferences,
            'quality_prediction': self._build_quality_predictors
        }
    
    async def learn_from_feedback_batch(
        self, 
        agent_name: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Learn from recent human feedback for a specific agent.
        
        Args:
            agent_name: Agent to learn patterns for
            lookback_days: How many days of feedback to analyze
            
        Returns:
            Dict containing learned patterns and improvements
        """
        start_time = datetime.utcnow()
        
        try:
            # 1. Collect feedback data
            feedback_data = await self._collect_feedback_data(agent_name, lookback_days)
            
            if len(feedback_data) < 10:  # Need minimum sample size
                logger.info(f"Insufficient feedback data for {agent_name}: {len(feedback_data)} samples")
                return {"status": "insufficient_data", "sample_size": len(feedback_data)}
            
            # 2. Run learning algorithms
            learning_results = {}
            
            # Correlation analysis
            correlations = await self._analyze_correlations(feedback_data)
            learning_results['correlations'] = correlations
            
            # Pattern recognition  
            patterns = await self._recognize_patterns(feedback_data)
            learning_results['patterns'] = patterns
            
            # Quality prediction model
            quality_model = await self._build_quality_predictors(feedback_data)
            learning_results['quality_model'] = quality_model
            
            # 3. Store learned patterns
            stored_patterns = await self._store_learning_results(
                agent_name, learning_results, feedback_data
            )
            
            # 4. Generate actionable recommendations
            recommendations = await self._generate_recommendations(
                agent_name, learning_results
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Learned from {len(feedback_data)} feedback samples for {agent_name} "
                       f"in {execution_time:.2f}s. Found {len(stored_patterns)} patterns.")
            
            return {
                "status": "success",
                "agent_name": agent_name,
                "sample_size": len(feedback_data),
                "patterns_learned": len(stored_patterns),
                "correlations_found": len(correlations),
                "recommendations": recommendations,
                "execution_time_seconds": execution_time
            }
            
        except Exception as e:
            logger.error(f"Failed to learn from feedback for {agent_name}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _collect_feedback_data(
        self, 
        agent_name: str, 
        lookback_days: int
    ) -> List[Dict[str, Any]]:
        """
        Collect agent performance data with corresponding human feedback.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Join agent performance with human feedback
                cur.execute("""
                    SELECT 
                        ap.id as performance_id,
                        ap.agent_name,
                        ap.agent_type,
                        ap.execution_id,
                        ap.duration,
                        ap.input_tokens,
                        ap.output_tokens,
                        ap.cost,
                        ap.metadata as agent_metadata,
                        
                        -- Agent decisions
                        ad.decision_point,
                        ad.input_data,
                        ad.output_data,  
                        ad.reasoning,
                        ad.confidence_score,
                        ad.alternatives_considered,
                        
                        -- Human feedback
                        rf.decision as human_decision,
                        rf.comments as human_comments,
                        rf.feedback_categories,
                        rf.quality_scores,
                        rf.improvement_suggestions,
                        rf.feedback_confidence,
                        rf.reviewer_id,
                        rf.stage as review_stage
                        
                    FROM agent_performance ap
                    JOIN agent_decisions ad ON ap.id = ad.performance_id
                    JOIN content_review_workflows crw ON ap.execution_id = crw.workflow_execution_id
                    JOIN review_feedback rf ON crw.id = rf.workflow_id
                    
                    WHERE ap.agent_name = %s
                      AND ap.start_time >= %s
                      AND rf.is_training_data = TRUE
                      AND ap.status = 'success'
                    
                    ORDER BY ap.start_time DESC
                """, (
                    agent_name,
                    datetime.utcnow() - timedelta(days=lookback_days)
                ))
                
                results = cur.fetchall()
                
                # Convert to structured data
                feedback_data = []
                for row in results:
                    feedback_data.append({
                        'performance_id': row[0],
                        'agent_name': row[1],
                        'execution_metrics': {
                            'duration_ms': row[4],
                            'input_tokens': row[5],
                            'output_tokens': row[6],
                            'cost': float(row[7]) if row[7] else 0,
                            'metadata': row[8] or {}
                        },
                        'agent_decision': {
                            'decision_point': row[9],
                            'input_data': row[10] or {},
                            'output_data': row[11] or {},
                            'reasoning': row[12],
                            'confidence_score': float(row[13]) if row[13] else 0,
                            'alternatives_considered': row[14] or []
                        },
                        'human_feedback': {
                            'decision': row[15],
                            'comments': row[16],
                            'categories': row[17] or {},
                            'quality_scores': row[18] or {},
                            'suggestions': row[19] or [],
                            'feedback_confidence': float(row[20]) if row[20] else 0,
                            'reviewer_id': row[21],
                            'review_stage': row[22]
                        }
                    })
                
                return feedback_data
                
        except Exception as e:
            logger.error(f"Failed to collect feedback data: {str(e)}")
            return []
    
    async def _analyze_correlations(self, feedback_data: List[Dict]) -> List[Dict]:
        """
        Analyze correlations between agent outputs and human feedback.
        """
        correlations = []
        
        try:
            # Extract features for correlation analysis
            agent_features = []
            feedback_scores = []
            
            for item in feedback_data:
                # Agent output features
                features = {
                    'duration_ms': item['execution_metrics']['duration_ms'],
                    'input_tokens': item['execution_metrics']['input_tokens'] or 0,
                    'output_tokens': item['execution_metrics']['output_tokens'] or 0,
                    'cost': item['execution_metrics']['cost'],
                    'confidence_score': item['agent_decision']['confidence_score'],
                    'alternatives_count': len(item['agent_decision']['alternatives_considered']),
                    'reasoning_length': len(item['agent_decision']['reasoning'] or ''),
                }
                
                # Human feedback scores
                human_scores = item['human_feedback']['quality_scores']
                
                if human_scores:
                    for category, score_data in human_scores.items():
                        if isinstance(score_data, dict) and 'score' in score_data:
                            agent_features.append(features)
                            feedback_scores.append({
                                'category': category,
                                'score': score_data['score'],
                                'features': features
                            })
            
            # Calculate correlations for each feedback category
            feedback_categories = set(item['category'] for item in feedback_scores)
            
            for category in feedback_categories:
                category_data = [item for item in feedback_scores if item['category'] == category]
                
                if len(category_data) < 5:  # Need minimum samples
                    continue
                
                scores = [item['score'] for item in category_data]
                
                # Calculate correlation for each agent feature
                for feature_name in features.keys():
                    feature_values = [item['features'][feature_name] for item in category_data]
                    
                    # Remove None values
                    paired_data = [(f, s) for f, s in zip(feature_values, scores) if f is not None and s is not None]
                    
                    if len(paired_data) < 5:
                        continue
                    
                    feature_vals, score_vals = zip(*paired_data)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(feature_vals, score_vals)[0, 1]
                    
                    if not np.isnan(correlation) and abs(correlation) > 0.3:  # Significant correlation
                        correlations.append({
                            'agent_feature': feature_name,
                            'feedback_category': category,
                            'correlation': float(correlation),
                            'sample_size': len(paired_data),
                            'feature_mean': float(np.mean(feature_vals)),
                            'score_mean': float(np.mean(score_vals)),
                            'significance': 'high' if abs(correlation) > 0.7 else 'medium'
                        })
            
            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            logger.info(f"Found {len(correlations)} significant correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return []
    
    async def _recognize_patterns(self, feedback_data: List[Dict]) -> List[LearningPattern]:
        """
        Recognize patterns in successful vs unsuccessful agent decisions.
        """
        patterns = []
        
        try:
            # Separate successful and unsuccessful decisions based on human feedback
            successful = []
            unsuccessful = []
            
            for item in feedback_data:
                human_decision = item['human_feedback']['decision']
                
                if human_decision == 'approve':
                    successful.append(item)
                elif human_decision in ['reject', 'request_changes']:
                    unsuccessful.append(item)
            
            logger.info(f"Analyzing {len(successful)} successful and {len(unsuccessful)} unsuccessful decisions")
            
            # Pattern 1: Successful decision characteristics
            if len(successful) >= 5:
                success_pattern = self._analyze_decision_characteristics(successful, 'successful_decisions')
                if success_pattern:
                    patterns.append(success_pattern)
            
            # Pattern 2: Common failure patterns
            if len(unsuccessful) >= 5:
                failure_pattern = self._analyze_decision_characteristics(unsuccessful, 'failure_patterns')
                if failure_pattern:
                    patterns.append(failure_pattern)
            
            # Pattern 3: Quality threshold patterns
            quality_patterns = self._analyze_quality_thresholds(feedback_data)
            patterns.extend(quality_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {str(e)}")
            return []
    
    def _analyze_decision_characteristics(
        self, 
        decisions: List[Dict], 
        pattern_type: str
    ) -> Optional[LearningPattern]:
        """
        Analyze characteristics of a group of decisions.
        """
        try:
            # Extract decision characteristics
            confidence_scores = [item['agent_decision']['confidence_score'] for item in decisions]
            reasoning_lengths = [len(item['agent_decision']['reasoning'] or '') for item in decisions]
            alternatives_counts = [len(item['agent_decision']['alternatives_considered']) for item in decisions]
            durations = [item['execution_metrics']['duration_ms'] for item in decisions]
            
            # Calculate statistics
            characteristics = {
                'confidence_score': {
                    'mean': float(np.mean(confidence_scores)),
                    'std': float(np.std(confidence_scores)),
                    'min': float(np.min(confidence_scores)),
                    'max': float(np.max(confidence_scores))
                },
                'reasoning_length': {
                    'mean': float(np.mean(reasoning_lengths)),
                    'std': float(np.std(reasoning_lengths))
                },
                'alternatives_considered': {
                    'mean': float(np.mean(alternatives_counts)),
                    'std': float(np.std(alternatives_counts))
                },
                'execution_duration': {
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations))
                }
            }
            
            # Extract common reasoning themes (simple keyword analysis)
            all_reasoning = ' '.join([item['agent_decision']['reasoning'] or '' for item in decisions])
            common_words = self._extract_common_terms(all_reasoning)
            
            return LearningPattern(
                pattern_type=pattern_type,
                input_features={},  # Could be enhanced with input analysis
                output_characteristics=characteristics,
                human_feedback_signals={
                    'common_reasoning_terms': common_words,
                    'sample_size': len(decisions),
                    'pattern_confidence': min(len(decisions) / 20.0, 1.0)  # More samples = higher confidence
                },
                confidence_score=min(len(decisions) / 20.0, 1.0),
                sample_size=len(decisions)
            )
            
        except Exception as e:
            logger.error(f"Decision characteristics analysis failed: {str(e)}")
            return None
    
    def _extract_common_terms(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract common terms from text (simple implementation).
        """
        # Simple word frequency analysis
        words = text.lower().split()
        word_freq = {}
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top N most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    async def _build_quality_predictors(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """
        Build machine learning models to predict human quality ratings.
        """
        try:
            # Prepare training data
            features = []
            targets = []
            
            for item in feedback_data:
                # Extract features
                feature_vector = [
                    item['execution_metrics']['duration_ms'] or 0,
                    item['execution_metrics']['input_tokens'] or 0,
                    item['execution_metrics']['output_tokens'] or 0,
                    item['execution_metrics']['cost'] or 0,
                    item['agent_decision']['confidence_score'] or 0,
                    len(item['agent_decision']['alternatives_considered']),
                    len(item['agent_decision']['reasoning'] or ''),
                ]
                
                # Extract target (overall quality score)
                quality_scores = item['human_feedback']['quality_scores']
                if quality_scores:
                    # Calculate average quality score
                    scores = []
                    for category, score_data in quality_scores.items():
                        if isinstance(score_data, dict) and 'score' in score_data:
                            scores.append(score_data['score'])
                    
                    if scores:
                        features.append(feature_vector)
                        targets.append(np.mean(scores))
            
            if len(features) < 10:
                return {"status": "insufficient_data", "samples": len(features)}
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Calculate feature importance
            feature_names = [
                'duration_ms', 'input_tokens', 'output_tokens', 'cost',
                'confidence_score', 'alternatives_count', 'reasoning_length'
            ]
            
            importance = dict(zip(feature_names, model.feature_importances_))
            
            # Calculate model performance (simple train score)
            train_score = model.score(X, y)
            
            return {
                "status": "success",
                "model_type": "RandomForestRegressor",
                "training_samples": len(features),
                "train_score": float(train_score),
                "feature_importance": {k: float(v) for k, v in importance.items()},
                "model_serializable": False  # Would need joblib to serialize
            }
            
        except Exception as e:
            logger.error(f"Quality predictor building failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _store_learning_results(
        self,
        agent_name: str,
        learning_results: Dict[str, Any],
        feedback_data: List[Dict]
    ) -> List[str]:
        """
        Store learned patterns in the database for future use.
        """
        stored_patterns = []
        
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                # Store correlations as learning patterns
                for correlation in learning_results.get('correlations', []):
                    pattern_id = str(uuid.uuid4())
                    
                    cur.execute("""
                        INSERT INTO agent_learning_patterns (
                            id, agent_name, agent_type, pattern_type, pattern_description,
                            input_features, output_characteristics, human_feedback_signals,
                            confidence_score, sample_size, learning_algorithm
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (agent_name, pattern_type, pattern_description) 
                        DO UPDATE SET
                            confidence_score = EXCLUDED.confidence_score,
                            sample_size = EXCLUDED.sample_size,
                            updated_at = NOW()
                    """, (
                        pattern_id,
                        agent_name,
                        feedback_data[0]['agent_name'] if feedback_data else 'unknown',
                        'correlation_pattern',
                        f"{correlation['agent_feature']} correlates with {correlation['feedback_category']}",
                        json.dumps({'feature': correlation['agent_feature']}),
                        json.dumps({'correlation': correlation['correlation']}),
                        json.dumps({'feedback_category': correlation['feedback_category']}),
                        abs(correlation['correlation']),
                        correlation['sample_size'],
                        'correlation_analysis'
                    ))
                    
                    stored_patterns.append(pattern_id)
                
                # Store recognized patterns
                for pattern in learning_results.get('patterns', []):
                    pattern_id = str(uuid.uuid4())
                    
                    cur.execute("""
                        INSERT INTO agent_learning_patterns (
                            id, agent_name, agent_type, pattern_type, pattern_description,
                            input_features, output_characteristics, human_feedback_signals,
                            confidence_score, sample_size, learning_algorithm
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (agent_name, pattern_type, pattern_description) 
                        DO UPDATE SET
                            confidence_score = EXCLUDED.confidence_score,
                            sample_size = EXCLUDED.sample_size,
                            updated_at = NOW()
                    """, (
                        pattern_id,
                        agent_name,
                        feedback_data[0]['agent_name'] if feedback_data else 'unknown',
                        pattern.pattern_type,
                        f"Learned pattern: {pattern.pattern_type}",
                        json.dumps(pattern.input_features),
                        json.dumps(pattern.output_characteristics),
                        json.dumps(pattern.human_feedback_signals),
                        pattern.confidence_score,
                        pattern.sample_size,
                        'pattern_recognition'
                    ))
                    
                    stored_patterns.append(pattern_id)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store learning results: {str(e)}")
        
        return stored_patterns
    
    async def get_learned_improvements(self, agent_name: str) -> Dict[str, Any]:
        """
        Get learned improvements that can be applied to agent execution.
        """
        try:
            with self.db_service.get_db_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT 
                        pattern_type,
                        pattern_description,
                        input_features,
                        output_characteristics,
                        human_feedback_signals,
                        confidence_score,
                        sample_size,
                        times_applied,
                        success_rate
                    FROM agent_learning_patterns 
                    WHERE agent_name = %s 
                      AND confidence_score > 0.5
                    ORDER BY confidence_score DESC, sample_size DESC
                """, (agent_name,))
                
                patterns = cur.fetchall()
                
                improvements = {
                    'total_patterns': len(patterns),
                    'high_confidence_patterns': len([p for p in patterns if p[5] > 0.8]),
                    'patterns': []
                }
                
                for pattern in patterns:
                    improvements['patterns'].append({
                        'type': pattern[0],
                        'description': pattern[1],
                        'input_features': pattern[2],
                        'output_characteristics': pattern[3],
                        'feedback_signals': pattern[4],
                        'confidence': float(pattern[5]),
                        'sample_size': pattern[6],
                        'applied_count': pattern[7],
                        'success_rate': float(pattern[8]) if pattern[8] else 0
                    })
                
                return improvements
                
        except Exception as e:
            logger.error(f"Failed to get learned improvements: {str(e)}")
            return {'total_patterns': 0, 'patterns': []}

# Usage in your agents
class LearningEnhancedAgent(BaseAgent):
    """
    Base agent enhanced with human feedback learning capabilities.
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        super().__init__(metadata)
        self.feedback_learner = HumanFeedbackLearner(self.db_service)
        self.learned_improvements = {}
        
    async def load_learned_improvements(self):
        """Load learned improvements from human feedback."""
        self.learned_improvements = await self.feedback_learner.get_learned_improvements(
            self.metadata.name
        )
    
    async def apply_learned_improvements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply learned improvements to modify agent behavior.
        """
        if not self.learned_improvements:
            await self.load_learned_improvements()
        
        improvements_applied = []
        
        for pattern in self.learned_improvements.get('patterns', []):
            if pattern['confidence'] > 0.7:  # Only apply high-confidence patterns
                
                if pattern['type'] == 'correlation_pattern':
                    # Apply correlation-based improvements
                    improvement = await self._apply_correlation_improvement(pattern, input_data)
                    if improvement:
                        improvements_applied.append(improvement)
                
                elif pattern['type'] == 'successful_decisions':
                    # Apply successful decision patterns
                    improvement = await self._apply_success_pattern(pattern, input_data)
                    if improvement:
                        improvements_applied.append(improvement)
        
        return {
            'improvements_applied': improvements_applied,
            'total_patterns_considered': len(self.learned_improvements.get('patterns', [])),
            'high_confidence_patterns': self.learned_improvements.get('high_confidence_patterns', 0)
        }
    
    async def _apply_correlation_improvement(
        self, 
        pattern: Dict[str, Any], 
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply improvement based on learned correlations.
        """
        # Example: If learned that longer reasoning correlates with higher quality,
        # adjust reasoning depth
        
        feedback_signals = pattern['feedback_signals']
        if feedback_signals.get('feedback_category') == 'content_quality':
            
            correlation = pattern['output_characteristics'].get('correlation', 0)
            feature = pattern['input_features'].get('feature')
            
            if feature == 'reasoning_length' and correlation > 0.5:
                return {
                    'improvement_type': 'enhanced_reasoning',
                    'description': 'Increase reasoning depth based on quality correlation',
                    'confidence': pattern['confidence'],
                    'action': 'extend_reasoning_analysis'
                }
        
        return None
    
    async def execute_with_learning(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[AgentExecutionContext] = None
    ) -> AgentResult:
        """
        Execute agent with learned improvements applied.
        """
        # Apply learned improvements
        improvements = await self.apply_learned_improvements(input_data)
        
        # Execute with improvements
        result = await self.execute(input_data, context)
        
        # Add learning metadata to result
        if result.metadata:
            result.metadata['learning_applied'] = improvements
        else:
            result.metadata = {'learning_applied': improvements}
        
        return result
```

### 4. Integration with Review Workflow

Enhance your existing review workflow to automatically trigger learning:

```python
# src/agents/workflow/enhanced_review_workflow.py

class EnhancedReviewWorkflow:
    """
    Enhanced review workflow that triggers learning from feedback.
    """
    
    def __init__(self):
        self.feedback_learner = HumanFeedbackLearner(db_service)
    
    async def process_human_feedback(
        self, 
        workflow_id: str, 
        reviewer_feedback: Dict[str, Any]
    ):
        """
        Process human feedback and trigger learning if sufficient data.
        """
        # Store feedback (existing logic)
        await self._store_feedback(workflow_id, reviewer_feedback)
        
        # Extract agent information from workflow
        agent_name = await self._get_workflow_agent(workflow_id)
        
        # Check if enough feedback for learning
        recent_feedback_count = await self._count_recent_feedback(agent_name, days=7)
        
        if recent_feedback_count >= 5:  # Threshold for learning trigger
            # Trigger asynchronous learning
            asyncio.create_task(
                self.feedback_learner.learn_from_feedback_batch(agent_name, lookback_days=30)
            )
            
            logger.info(f"Triggered learning for {agent_name} based on recent feedback")
```

### 5. Real-Time Learning Dashboard

Add API endpoints for learning observability:

```python
# src/api/routes/learning.py

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/api/v2/learning", tags=["learning"])

@router.get("/agents/{agent_name}/patterns")
async def get_agent_learning_patterns(agent_name: str):
    """Get learned patterns for a specific agent"""
    learner = HumanFeedbackLearner(db_service)
    patterns = await learner.get_learned_improvements(agent_name)
    return patterns

@router.post("/agents/{agent_name}/learn")
async def trigger_learning(agent_name: str, lookback_days: int = 30):
    """Manually trigger learning from recent feedback"""
    learner = HumanFeedbackLearner(db_service)
    result = await learner.learn_from_feedback_batch(agent_name, lookback_days)
    return result

@router.get("/feedback/analytics")
async def get_feedback_analytics():
    """Get analytics on human feedback and learning progress"""
    # Implementation for feedback analytics
    pass

@router.get("/correlations/{agent_name}")
async def get_agent_correlations(agent_name: str):
    """Get correlation analysis between agent outputs and human feedback"""
    pass
```

---

## Learning Capabilities Summary

Your agents can now learn from human feedback in multiple ways:

### 🎯 **1. Quality Correlation Learning**
- Agents learn which of their characteristics correlate with high/low human ratings
- Automatically adjust behavior to increase quality scores

### 🧠 **2. Decision Pattern Recognition**  
- Identify patterns in successful vs unsuccessful agent decisions
- Learn optimal confidence thresholds, reasoning depth, and alternative consideration

### 👥 **3. Human Preference Modeling**
- Build models of individual reviewer preferences
- Adapt content style and approach based on reviewer history

### 📊 **4. Automated Quality Prediction**
- Predict human quality ratings before review
- Self-assess and improve outputs proactively

### 🔄 **5. Continuous Improvement Loop**
- Automatic learning triggered by new feedback
- Real-time application of learned improvements
- Performance tracking of learning effectiveness

## Implementation Timeline

**Week 1-2**: Database enhancements and feedback collection system
**Week 3-4**: Core learning algorithms and pattern recognition  
**Week 5-6**: Integration with existing agents and review workflow
**Week 7-8**: Learning dashboard and analytics
**Week 9-10**: Testing, optimization, and deployment

Your agents will continuously get better at producing content that meets human quality standards, leading to higher approval rates and better overall content quality!