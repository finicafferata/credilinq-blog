"""
Advanced Checkpoint and Recovery System for LangGraph Workflows

Provides sophisticated checkpointing, state persistence, recovery mechanisms,
and fault tolerance for complex multi-agent workflows.
"""

import json
import pickle
import asyncio
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CheckpointType(Enum):
    STATE_SNAPSHOT = "state_snapshot"
    AGENT_EXECUTION = "agent_execution"
    WORKFLOW_MILESTONE = "workflow_milestone"
    ERROR_RECOVERY = "error_recovery"
    USER_INTERVENTION = "user_intervention"


class RecoveryStrategy(Enum):
    RETRY_FROM_LAST = "retry_from_last"
    RESTART_WORKFLOW = "restart_workflow"
    SKIP_TO_NEXT = "skip_to_next"
    MANUAL_INTERVENTION = "manual_intervention"
    ROLLBACK_TO_SAFE = "rollback_to_safe"


@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    workflow_id: str
    run_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    node_id: str
    state_hash: str
    execution_context: Dict[str, Any] = field(default_factory=dict)
    recovery_metadata: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    error_context: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class RecoveryPlan:
    strategy: RecoveryStrategy
    target_checkpoint_id: Optional[str] = None
    skip_nodes: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    recovery_actions: List[str] = field(default_factory=list)
    human_intervention_required: bool = False
    estimated_recovery_time: Optional[int] = None


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends"""
    
    @abstractmethod
    async def save_checkpoint(self, checkpoint_id: str, state_data: bytes, 
                            metadata: CheckpointMetadata) -> None:
        pass
    
    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[tuple[bytes, CheckpointMetadata]]:
        pass
    
    @abstractmethod
    async def list_checkpoints(self, workflow_id: str, run_id: Optional[str] = None) -> List[CheckpointMetadata]:
        pass
    
    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        pass
    
    @abstractmethod
    async def cleanup_old_checkpoints(self, retention_hours: int = 72) -> int:
        pass


class FileSystemCheckpointStorage(CheckpointStorage):
    """File system-based checkpoint storage implementation"""
    
    def __init__(self, base_path: Union[str, Path] = "./.langgraph_checkpoints"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    async def save_checkpoint(self, checkpoint_id: str, state_data: bytes, 
                            metadata: CheckpointMetadata) -> None:
        """Save checkpoint to file system"""
        checkpoint_dir = self.base_path / metadata.workflow_id / metadata.run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state data
        state_file = checkpoint_dir / f"{checkpoint_id}.state"
        async with asyncio.to_thread(open, state_file, 'wb') as f:
            await asyncio.to_thread(f.write, state_data)
        
        # Save metadata
        metadata_file = checkpoint_dir / f"{checkpoint_id}.metadata.json"
        metadata_dict = {
            "checkpoint_id": metadata.checkpoint_id,
            "workflow_id": metadata.workflow_id,
            "run_id": metadata.run_id,
            "checkpoint_type": metadata.checkpoint_type.value,
            "created_at": metadata.created_at.isoformat(),
            "node_id": metadata.node_id,
            "state_hash": metadata.state_hash,
            "execution_context": metadata.execution_context,
            "recovery_metadata": metadata.recovery_metadata,
            "agent_states": metadata.agent_states,
            "quality_scores": metadata.quality_scores,
            "error_context": metadata.error_context,
            "dependencies": metadata.dependencies,
            "size_bytes": metadata.size_bytes
        }
        
        async with asyncio.to_thread(open, metadata_file, 'w') as f:
            await asyncio.to_thread(json.dump, metadata_dict, f, indent=2)
        
        logger.info(f"Saved checkpoint {checkpoint_id} for workflow {metadata.workflow_id}")
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[tuple[bytes, CheckpointMetadata]]:
        """Load checkpoint from file system"""
        # Find the checkpoint file (search all workflow/run directories)
        for workflow_dir in self.base_path.iterdir():
            if not workflow_dir.is_dir():
                continue
            for run_dir in workflow_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                state_file = run_dir / f"{checkpoint_id}.state"
                metadata_file = run_dir / f"{checkpoint_id}.metadata.json"
                
                if state_file.exists() and metadata_file.exists():
                    # Load state data
                    async with asyncio.to_thread(open, state_file, 'rb') as f:
                        state_data = await asyncio.to_thread(f.read)
                    
                    # Load metadata
                    async with asyncio.to_thread(open, metadata_file, 'r') as f:
                        metadata_dict = await asyncio.to_thread(json.load, f)
                    
                    metadata = CheckpointMetadata(
                        checkpoint_id=metadata_dict["checkpoint_id"],
                        workflow_id=metadata_dict["workflow_id"],
                        run_id=metadata_dict["run_id"],
                        checkpoint_type=CheckpointType(metadata_dict["checkpoint_type"]),
                        created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                        node_id=metadata_dict["node_id"],
                        state_hash=metadata_dict["state_hash"],
                        execution_context=metadata_dict.get("execution_context", {}),
                        recovery_metadata=metadata_dict.get("recovery_metadata", {}),
                        agent_states=metadata_dict.get("agent_states", {}),
                        quality_scores=metadata_dict.get("quality_scores", {}),
                        error_context=metadata_dict.get("error_context"),
                        dependencies=metadata_dict.get("dependencies", []),
                        size_bytes=metadata_dict.get("size_bytes", 0)
                    )
                    
                    return state_data, metadata
        
        return None
    
    async def list_checkpoints(self, workflow_id: str, run_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List checkpoints for a workflow/run"""
        checkpoints = []
        workflow_dir = self.base_path / workflow_id
        
        if not workflow_dir.exists():
            return checkpoints
        
        run_dirs = [workflow_dir / run_id] if run_id else workflow_dir.iterdir()
        
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            
            for metadata_file in run_dir.glob("*.metadata.json"):
                try:
                    async with asyncio.to_thread(open, metadata_file, 'r') as f:
                        metadata_dict = await asyncio.to_thread(json.load, f)
                    
                    metadata = CheckpointMetadata(
                        checkpoint_id=metadata_dict["checkpoint_id"],
                        workflow_id=metadata_dict["workflow_id"],
                        run_id=metadata_dict["run_id"],
                        checkpoint_type=CheckpointType(metadata_dict["checkpoint_type"]),
                        created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                        node_id=metadata_dict["node_id"],
                        state_hash=metadata_dict["state_hash"],
                        execution_context=metadata_dict.get("execution_context", {}),
                        recovery_metadata=metadata_dict.get("recovery_metadata", {}),
                        agent_states=metadata_dict.get("agent_states", {}),
                        quality_scores=metadata_dict.get("quality_scores", {}),
                        error_context=metadata_dict.get("error_context"),
                        dependencies=metadata_dict.get("dependencies", []),
                        size_bytes=metadata_dict.get("size_bytes", 0)
                    )
                    
                    checkpoints.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint metadata from {metadata_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.created_at, reverse=True)
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        checkpoint_data = await self.load_checkpoint(checkpoint_id)
        if not checkpoint_data:
            return False
        
        _, metadata = checkpoint_data
        checkpoint_dir = self.base_path / metadata.workflow_id / metadata.run_id
        
        state_file = checkpoint_dir / f"{checkpoint_id}.state"
        metadata_file = checkpoint_dir / f"{checkpoint_id}.metadata.json"
        
        try:
            if state_file.exists():
                await asyncio.to_thread(state_file.unlink)
            if metadata_file.exists():
                await asyncio.to_thread(metadata_file.unlink)
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, retention_hours: int = 72) -> int:
        """Clean up checkpoints older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        deleted_count = 0
        
        for workflow_dir in self.base_path.iterdir():
            if not workflow_dir.is_dir():
                continue
            for run_dir in workflow_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                for metadata_file in run_dir.glob("*.metadata.json"):
                    try:
                        async with asyncio.to_thread(open, metadata_file, 'r') as f:
                            metadata_dict = await asyncio.to_thread(json.load, f)
                        
                        created_at = datetime.fromisoformat(metadata_dict["created_at"])
                        if created_at < cutoff_time:
                            checkpoint_id = metadata_dict["checkpoint_id"]
                            if await self.delete_checkpoint(checkpoint_id):
                                deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error during cleanup of {metadata_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count


class WorkflowCheckpointer(Generic[T]):
    """Advanced checkpointing system for LangGraph workflows"""
    
    def __init__(self, storage: CheckpointStorage, 
                 auto_checkpoint: bool = True,
                 checkpoint_frequency: int = 5,
                 max_checkpoints_per_run: int = 50):
        self.storage = storage
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints_per_run = max_checkpoints_per_run
        self.checkpoint_counters: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable[[Exception, CheckpointMetadata], RecoveryPlan]] = {}
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common error types"""
        
        def timeout_recovery(error: Exception, last_checkpoint: CheckpointMetadata) -> RecoveryPlan:
            return RecoveryPlan(
                strategy=RecoveryStrategy.RETRY_FROM_LAST,
                target_checkpoint_id=last_checkpoint.checkpoint_id,
                max_retries=2,
                recovery_actions=["increase_timeout", "optimize_agent_execution"]
            )
        
        def validation_error_recovery(error: Exception, last_checkpoint: CheckpointMetadata) -> RecoveryPlan:
            return RecoveryPlan(
                strategy=RecoveryStrategy.ROLLBACK_TO_SAFE,
                target_checkpoint_id=last_checkpoint.checkpoint_id,
                recovery_actions=["validate_inputs", "sanitize_data"]
            )
        
        def llm_error_recovery(error: Exception, last_checkpoint: CheckpointMetadata) -> RecoveryPlan:
            return RecoveryPlan(
                strategy=RecoveryStrategy.RETRY_FROM_LAST,
                target_checkpoint_id=last_checkpoint.checkpoint_id,
                max_retries=3,
                recovery_actions=["switch_llm_provider", "reduce_complexity"]
            )
        
        self.recovery_strategies.update({
            "TimeoutError": timeout_recovery,
            "ValidationError": validation_error_recovery,
            "LLMError": llm_error_recovery,
            "ConnectionError": timeout_recovery
        })
    
    def register_recovery_strategy(self, error_type: str, 
                                 strategy_func: Callable[[Exception, CheckpointMetadata], RecoveryPlan]):
        """Register custom recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy_func
    
    async def create_checkpoint(self, workflow_id: str, run_id: str, 
                              current_state: T, node_id: str,
                              checkpoint_type: CheckpointType = CheckpointType.STATE_SNAPSHOT,
                              execution_context: Optional[Dict[str, Any]] = None,
                              agent_states: Optional[Dict[str, Any]] = None,
                              quality_scores: Optional[Dict[str, float]] = None) -> str:
        """Create a new checkpoint"""
        
        # Generate checkpoint ID
        timestamp = datetime.utcnow()
        checkpoint_id = f"{workflow_id}_{run_id}_{node_id}_{int(timestamp.timestamp())}"
        
        # Serialize state
        if isinstance(current_state, dict):
            state_data = json.dumps(current_state, default=str).encode('utf-8')
        else:
            state_data = pickle.dumps(current_state)
        
        # Calculate state hash for integrity checking
        state_hash = hashlib.sha256(state_data).hexdigest()
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            run_id=run_id,
            checkpoint_type=checkpoint_type,
            created_at=timestamp,
            node_id=node_id,
            state_hash=state_hash,
            execution_context=execution_context or {},
            agent_states=agent_states or {},
            quality_scores=quality_scores or {},
            size_bytes=len(state_data)
        )
        
        # Save checkpoint
        await self.storage.save_checkpoint(checkpoint_id, state_data, metadata)
        
        # Update counter
        counter_key = f"{workflow_id}_{run_id}"
        self.checkpoint_counters[counter_key] = self.checkpoint_counters.get(counter_key, 0) + 1
        
        # Cleanup old checkpoints if we exceed max
        if self.checkpoint_counters[counter_key] > self.max_checkpoints_per_run:
            await self._cleanup_old_checkpoints(workflow_id, run_id)
        
        logger.info(f"Created checkpoint {checkpoint_id} for node {node_id}")
        return checkpoint_id
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[tuple[T, CheckpointMetadata]]:
        """Restore state from a specific checkpoint"""
        checkpoint_data = await self.storage.load_checkpoint(checkpoint_id)
        if not checkpoint_data:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return None
        
        state_data, metadata = checkpoint_data
        
        # Verify state integrity
        current_hash = hashlib.sha256(state_data).hexdigest()
        if current_hash != metadata.state_hash:
            logger.error(f"Checkpoint {checkpoint_id} integrity check failed")
            return None
        
        # Deserialize state
        try:
            if state_data.startswith(b'{') or state_data.startswith(b'['):
                # JSON data
                state = json.loads(state_data.decode('utf-8'))
            else:
                # Pickle data
                state = pickle.loads(state_data)
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint {checkpoint_id}: {e}")
            return None
        
        logger.info(f"Restored state from checkpoint {checkpoint_id}")
        return state, metadata
    
    async def get_recovery_plan(self, error: Exception, workflow_id: str, run_id: str) -> Optional[RecoveryPlan]:
        """Generate recovery plan based on error and available checkpoints"""
        
        # Get latest checkpoint for this run
        checkpoints = await self.storage.list_checkpoints(workflow_id, run_id)
        if not checkpoints:
            return RecoveryPlan(
                strategy=RecoveryStrategy.RESTART_WORKFLOW,
                recovery_actions=["restart_from_beginning"]
            )
        
        latest_checkpoint = checkpoints[0]
        error_type = type(error).__name__
        
        # Use registered recovery strategy if available
        if error_type in self.recovery_strategies:
            recovery_plan = self.recovery_strategies[error_type](error, latest_checkpoint)
        else:
            # Default recovery strategy
            recovery_plan = RecoveryPlan(
                strategy=RecoveryStrategy.RETRY_FROM_LAST,
                target_checkpoint_id=latest_checkpoint.checkpoint_id,
                recovery_actions=["generic_retry"]
            )
        
        # Add error context to the plan
        recovery_plan.recovery_actions.append(f"handle_{error_type.lower()}")
        
        logger.info(f"Generated recovery plan for {error_type}: {recovery_plan.strategy.value}")
        return recovery_plan
    
    async def execute_recovery(self, recovery_plan: RecoveryPlan) -> Optional[tuple[T, CheckpointMetadata]]:
        """Execute a recovery plan"""
        
        if recovery_plan.strategy == RecoveryStrategy.RETRY_FROM_LAST:
            if recovery_plan.target_checkpoint_id:
                return await self.restore_from_checkpoint(recovery_plan.target_checkpoint_id)
        
        elif recovery_plan.strategy == RecoveryStrategy.ROLLBACK_TO_SAFE:
            # Find the most recent "safe" checkpoint (milestone type)
            if recovery_plan.target_checkpoint_id:
                checkpoint_data = await self.storage.load_checkpoint(recovery_plan.target_checkpoint_id)
                if checkpoint_data:
                    _, metadata = checkpoint_data
                    safe_checkpoints = await self.storage.list_checkpoints(metadata.workflow_id, metadata.run_id)
                    for checkpoint in safe_checkpoints:
                        if checkpoint.checkpoint_type == CheckpointType.WORKFLOW_MILESTONE:
                            return await self.restore_from_checkpoint(checkpoint.checkpoint_id)
        
        elif recovery_plan.strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            logger.warning("Manual intervention required for recovery")
            return None
        
        elif recovery_plan.strategy == RecoveryStrategy.RESTART_WORKFLOW:
            logger.info("Recovery requires workflow restart")
            return None
        
        logger.warning(f"Unknown recovery strategy: {recovery_plan.strategy}")
        return None
    
    async def should_create_checkpoint(self, workflow_id: str, run_id: str, node_id: str) -> bool:
        """Determine if a checkpoint should be created based on configuration"""
        if not self.auto_checkpoint:
            return False
        
        counter_key = f"{workflow_id}_{run_id}"
        current_count = self.checkpoint_counters.get(counter_key, 0)
        
        return current_count % self.checkpoint_frequency == 0
    
    async def get_checkpoint_history(self, workflow_id: str, run_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """Get checkpoint history for analysis"""
        return await self.storage.list_checkpoints(workflow_id, run_id)
    
    async def _cleanup_old_checkpoints(self, workflow_id: str, run_id: str, keep_count: int = 10):
        """Clean up old checkpoints, keeping only the most recent ones"""
        checkpoints = await self.storage.list_checkpoints(workflow_id, run_id)
        
        if len(checkpoints) > keep_count:
            # Keep milestone checkpoints and recent ones
            to_keep = []
            regular_checkpoints = []
            
            for checkpoint in checkpoints:
                if checkpoint.checkpoint_type == CheckpointType.WORKFLOW_MILESTONE:
                    to_keep.append(checkpoint)
                else:
                    regular_checkpoints.append(checkpoint)
            
            # Keep most recent regular checkpoints
            to_keep.extend(regular_checkpoints[:keep_count - len(to_keep)])
            
            # Delete the rest
            to_delete = [cp for cp in checkpoints if cp not in to_keep]
            for checkpoint in to_delete:
                await self.storage.delete_checkpoint(checkpoint.checkpoint_id)
            
            logger.info(f"Cleaned up {len(to_delete)} old checkpoints for {workflow_id}/{run_id}")


# Global checkpoint system
checkpoint_storage = FileSystemCheckpointStorage()
workflow_checkpointer = WorkflowCheckpointer(checkpoint_storage)


class CheckpointContextManager:
    """Context manager for automatic checkpointing"""
    
    def __init__(self, workflow_id: str, run_id: str, node_id: str,
                 checkpoint_type: CheckpointType = CheckpointType.STATE_SNAPSHOT):
        self.workflow_id = workflow_id
        self.run_id = run_id
        self.node_id = node_id
        self.checkpoint_type = checkpoint_type
        self.checkpoint_id: Optional[str] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success - create checkpoint if configured
            if await workflow_checkpointer.should_create_checkpoint(
                self.workflow_id, self.run_id, self.node_id
            ):
                self.checkpoint_id = await workflow_checkpointer.create_checkpoint(
                    self.workflow_id, self.run_id, {}, self.node_id, self.checkpoint_type
                )
        else:
            # Error occurred - create error recovery checkpoint
            self.checkpoint_id = await workflow_checkpointer.create_checkpoint(
                self.workflow_id, self.run_id, {},
                self.node_id, CheckpointType.ERROR_RECOVERY,
                execution_context={"error_type": exc_type.__name__, "error_message": str(exc_val)}
            )


# Convenience functions for checkpoint management
async def create_milestone_checkpoint(workflow_id: str, run_id: str, state: Any, node_id: str) -> str:
    """Create a milestone checkpoint for important workflow states"""
    return await workflow_checkpointer.create_checkpoint(
        workflow_id, run_id, state, node_id, CheckpointType.WORKFLOW_MILESTONE
    )


async def recover_workflow(workflow_id: str, run_id: str, error: Exception) -> Optional[tuple[Any, CheckpointMetadata]]:
    """Attempt to recover a failed workflow"""
    recovery_plan = await workflow_checkpointer.get_recovery_plan(error, workflow_id, run_id)
    if recovery_plan:
        return await workflow_checkpointer.execute_recovery(recovery_plan)
    return None