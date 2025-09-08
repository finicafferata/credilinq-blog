"""
Campaign Workflow Management Routes
Handles active workflows, workflow status, and WebSocket connections.
"""

import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import asyncio

from src.config.database import db_config

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/workflow/active", response_model=List[Dict[str, Any]]) 
async def get_active_campaign_workflows():
    """
    Get all currently active campaign workflows for Master Planner Dashboard visibility.
    """
    try:
        logger.info("Retrieving active campaign workflows")
        
        # Query database for campaigns with active agent execution
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    c.id,
                    c.name,
                    c.created_at,
                    c.metadata,
                    COUNT(ct.id) as total_tasks,
                    COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.metadata->>'processing_status' = 'generating_content'
                   OR EXISTS (
                       SELECT 1 FROM campaign_tasks ct2 
                       WHERE ct2.campaign_id = c.id 
                       AND ct2.status IN ('pending', 'running')
                   )
                GROUP BY c.id, c.name, c.created_at, c.metadata
                ORDER BY c.created_at DESC
            """)
            
            active_campaigns = cur.fetchall()
            
        workflows = []
        for campaign in active_campaigns:
            campaign_id, name, created_at, metadata, total_tasks, completed_tasks = campaign
            
            # Calculate progress
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Determine status based on metadata and task completion
            processing_status = metadata.get('processing_status') if metadata else None
            if processing_status == 'generating_content':
                status = 'running'
            elif completed_tasks == total_tasks and total_tasks > 0:
                status = 'completed' 
            else:
                status = 'pending'
            
            workflows.append({
                "workflow_execution_id": f"campaign_{campaign_id}",
                "campaign_id": campaign_id,
                "campaign_name": name,
                "status": status,
                "progress_percentage": int(progress),
                "total_agents": total_tasks,
                "completed_agents": completed_tasks,
                "start_time": created_at.isoformat() if created_at else None,
                "workflow_type": "campaign_content_generation",
                "last_heartbeat": datetime.now().isoformat()
            })
            
        logger.info(f"Found {len(workflows)} active campaign workflows")
        return workflows
        
    except Exception as e:
        logger.error(f"❌ Failed to get active campaign workflows: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active campaign workflows: {str(e)}"
        )

@router.get("/{campaign_id}/workflow-status")
async def get_campaign_workflow_status(campaign_id: str):
    """
    Get detailed workflow status for a specific campaign for Master Planner Dashboard.
    """
    try:
        logger.info(f"Retrieving workflow status for campaign: {campaign_id}")
        
        # Get campaign tasks and their status
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Get campaign info and tasks
            cur.execute("""
                SELECT 
                    c.name,
                    c.metadata,
                    c.created_at,
                    ct.task_type,
                    ct.status,
                    ct.agent_type,
                    ct.created_at as task_created,
                    ct.updated_at as task_updated,
                    ct.result
                FROM campaigns c
                LEFT JOIN campaign_tasks ct ON c.id = ct.campaign_id
                WHERE c.id = %s
                ORDER BY ct.created_at ASC
            """, (campaign_id,))
            
            results = cur.fetchall()
            
        if not results:
            raise HTTPException(status_code=404, detail="Campaign not found")
            
        # Process results
        campaign_name = results[0][0] if results[0][0] else "Unknown Campaign"
        metadata = results[0][1] or {}
        campaign_created = results[0][2]
        
        # Group agents by their execution status
        agents_status = {
            "waiting": [],
            "running": [],
            "completed": [],
            "failed": []
        }
        
        total_tasks = 0
        completed_tasks = 0
        
        for row in results:
            if row[3]:  # if task_type exists
                total_tasks += 1
                task_type, status, agent_type, task_created, task_updated, result_data = row[3:9]
                
                agent_info = {
                    "agent_type": agent_type or task_type,
                    "task_type": task_type,
                    "start_time": task_created.isoformat() if task_created else None,
                    "updated_time": task_updated.isoformat() if task_updated else None,
                    "output_preview": str(result_data)[:200] + "..." if result_data else None
                }
                
                if status == 'completed':
                    agents_status["completed"].append(agent_info)
                    completed_tasks += 1
                elif status == 'running':
                    agents_status["running"].append(agent_info)
                elif status == 'failed':
                    agents_status["failed"].append(agent_info) 
                else:
                    agents_status["waiting"].append(agent_info)
        
        # Calculate progress
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Determine overall status
        processing_status = metadata.get('processing_status')
        if processing_status == 'generating_content' and agents_status["running"]:
            status = 'running'
        elif completed_tasks == total_tasks and total_tasks > 0:
            status = 'completed'
        elif agents_status["failed"]:
            status = 'failed'
        else:
            status = 'waiting'
        
        return {
            "workflow_execution_id": f"campaign_{campaign_id}",
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "status": status,
            "progress_percentage": int(progress),
            "current_step": completed_tasks + len(agents_status["running"]),
            "total_steps": total_tasks,
            "agents_status": agents_status,
            "start_time": campaign_created.isoformat() if campaign_created else None,
            "last_heartbeat": datetime.now().isoformat(),
            "intermediate_outputs": {
                "total_tasks": total_tasks,
                "processing_status": processing_status,
                "workflow_type": "campaign_content_generation"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get campaign workflow status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


# WebSocket Connection Manager for Real-Time Updates
class CampaignWebSocketManager:
    """Enhanced WebSocket manager with robust connection handling and cleanup"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.connection_metadata: Dict[str, Dict] = {}  # Track connection timestamps and health
        
    async def connect(self, websocket: WebSocket, campaign_id: str):
        """Connect websocket with enhanced error handling and metadata tracking"""
        try:
            await websocket.accept()
            
            if campaign_id not in self.active_connections:
                self.active_connections[campaign_id] = []
                
            self.active_connections[campaign_id].append(websocket)
            
            # Track connection metadata
            connection_key = f"{campaign_id}_{id(websocket)}"
            self.connection_metadata[connection_key] = {
                "campaign_id": campaign_id,
                "connected_at": datetime.now(),
                "last_ping": datetime.now()
            }
            
            logger.info(f"📡 [WEBSOCKET] Client connected to campaign: {campaign_id} (Total: {len(self.active_connections[campaign_id])})")
            
        except Exception as e:
            logger.error(f"📡 [WEBSOCKET] Failed to accept connection for campaign {campaign_id}: {e}")
            raise
        
    def disconnect(self, websocket: WebSocket, campaign_id: str):
        """Safely disconnect websocket with proper cleanup"""
        try:
            if campaign_id in self.active_connections:
                if websocket in self.active_connections[campaign_id]:
                    self.active_connections[campaign_id].remove(websocket)
                    
                # Clean up empty campaign connections
                if not self.active_connections[campaign_id]:
                    del self.active_connections[campaign_id]
            
            # Clean up connection metadata
            connection_key = f"{campaign_id}_{id(websocket)}"
            if connection_key in self.connection_metadata:
                del self.connection_metadata[connection_key]
                
            logger.info(f"📡 [WEBSOCKET] Client disconnected from campaign: {campaign_id}")
            
        except Exception as e:
            logger.warning(f"📡 [WEBSOCKET] Error during disconnect cleanup: {e}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message with error handling and connection validation"""
        try:
            if websocket.client_state.value == 1:  # WebSocket.OPEN
                await websocket.send_text(message)
                return True
            else:
                logger.warning("📡 [WEBSOCKET] Attempted to send to closed connection")
                return False
        except Exception as e:
            logger.error(f"📡 [WEBSOCKET] Failed to send personal message: {e}")
            return False
        
    async def broadcast_to_campaign(self, message: Dict[str, Any], campaign_id: str):
        """Enhanced broadcast with connection health checks and cleanup"""
        if campaign_id not in self.active_connections:
            logger.debug(f"📡 [WEBSOCKET] No connections for campaign: {campaign_id}")
            return
            
        connections = self.active_connections[campaign_id].copy()  # Work with copy to avoid modification during iteration
        message_text = json.dumps(message, default=str)  # Handle datetime serialization
        disconnected_sockets = []
        successful_sends = 0
        
        for connection in connections:
            try:
                # Check if connection is still open
                if connection.client_state.value != 1:  # Not WebSocket.OPEN
                    disconnected_sockets.append(connection)
                    continue
                    
                await connection.send_text(message_text)
                successful_sends += 1
                
                # Update last ping for this connection
                connection_key = f"{campaign_id}_{id(connection)}"
                if connection_key in self.connection_metadata:
                    self.connection_metadata[connection_key]["last_ping"] = datetime.now()
                    
            except Exception as e:
                logger.warning(f"📡 [WEBSOCKET] Failed to send message to connection: {e}")
                disconnected_sockets.append(connection)
        
        # Clean up disconnected sockets
        for socket in disconnected_sockets:
            self.disconnect(socket, campaign_id)
            
        logger.info(f"📡 [WEBSOCKET] Broadcasted to campaign {campaign_id}: {message.get('type', 'unknown')} - {successful_sends}/{len(connections)} successful sends")
    
    async def cleanup_stale_connections(self):
        """Remove stale connections that haven't been active recently"""
        try:
            stale_threshold = datetime.now() - timedelta(minutes=30)
            stale_connections = []
            
            for connection_key, metadata in self.connection_metadata.items():
                if metadata["last_ping"] < stale_threshold:
                    stale_connections.append((connection_key, metadata))
            
            for connection_key, metadata in stale_connections:
                campaign_id = metadata["campaign_id"]
                # Find and disconnect the stale connection
                if campaign_id in self.active_connections:
                    for websocket in self.active_connections[campaign_id]:
                        if f"{campaign_id}_{id(websocket)}" == connection_key:
                            self.disconnect(websocket, campaign_id)
                            break
                            
            if stale_connections:
                logger.info(f"📡 [WEBSOCKET] Cleaned up {len(stale_connections)} stale connections")
                
        except Exception as e:
            logger.error(f"📡 [WEBSOCKET] Error during connection cleanup: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        return {
            "total_connections": total_connections,
            "campaigns_with_connections": len(self.active_connections),
            "connections_by_campaign": {
                campaign_id: len(connections) 
                for campaign_id, connections in self.active_connections.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = CampaignWebSocketManager()

@router.websocket("/ws/campaign/{campaign_id}/status")
async def websocket_campaign_status(websocket: WebSocket, campaign_id: str):
    """
    Enhanced WebSocket endpoint for real-time campaign status updates
    URL: /ws/campaign/{campaign_id}/status
    Features: Auto-reconnect support, heartbeat, robust error handling
    """
    connection_id = str(uuid.uuid4())[:8]
    logger.info(f"📡 [WEBSOCKET] New connection attempt for campaign: {campaign_id} (ID: {connection_id})")
    
    try:
        await websocket_manager.connect(websocket, campaign_id)
        
        # Send initial connection confirmation with enhanced data
        initial_message = {
            "type": "connection_established",
            "campaign_id": campaign_id,
            "connection_id": connection_id,
            "message": "Connected to real-time campaign updates",
            "timestamp": datetime.now().isoformat(),
            "server_time": datetime.now().isoformat()
        }
        
        success = await websocket_manager.send_personal_message(
            json.dumps(initial_message, default=str),
            websocket
        )
        
        if not success:
            logger.error(f"📡 [WEBSOCKET] Failed to send initial message to {connection_id}")
            return
            
        logger.info(f"📡 [WEBSOCKET] Connection established for campaign: {campaign_id} (ID: {connection_id})")
        
        # Keep connection alive and handle incoming messages
        heartbeat_interval = 30  # Send heartbeat every 30 seconds
        last_heartbeat = time.time()
        
        while True:
            try:
                # Set timeout for receive to allow periodic heartbeats
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                
                # Parse and handle client message
                try:
                    client_message = json.loads(data)
                    message_type = client_message.get("type", "unknown")
                    
                    if message_type == "ping":
                        # Respond to client ping with pong
                        pong_response = {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                            "connection_id": connection_id
                        }
                        await websocket_manager.send_personal_message(
                            json.dumps(pong_response, default=str),
                            websocket
                        )
                    else:
                        # Echo other messages back for testing
                        echo_response = {
                            "type": "echo",
                            "received": data,
                            "message_type": message_type,
                            "timestamp": datetime.now().isoformat(),
                            "connection_id": connection_id
                        }
                        await websocket_manager.send_personal_message(
                            json.dumps(echo_response, default=str),
                            websocket
                        )
                        
                except json.JSONDecodeError as e:
                    # Handle malformed JSON from client
                    error_response = {
                        "type": "error",
                        "error": "Invalid JSON format",
                        "received_data": data[:100] + "..." if len(data) > 100 else data,
                        "timestamp": datetime.now().isoformat(),
                        "connection_id": connection_id
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(error_response, default=str),
                        websocket
                    )
                    logger.warning(f"📡 [WEBSOCKET] Invalid JSON from client {connection_id}: {e}")
                
            except asyncio.TimeoutError:
                # Handle timeout - send heartbeat if needed
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    heartbeat_message = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat(),
                        "connection_id": connection_id,
                        "uptime_seconds": int(current_time - last_heartbeat)
                    }
                    
                    success = await websocket_manager.send_personal_message(
                        json.dumps(heartbeat_message, default=str),
                        websocket
                    )
                    
                    if success:
                        last_heartbeat = current_time
                    else:
                        logger.warning(f"📡 [WEBSOCKET] Heartbeat failed for {connection_id}")
                        break
                        
            except WebSocketDisconnect:
                logger.info(f"📡 [WEBSOCKET] Client {connection_id} disconnected normally from campaign: {campaign_id}")
                break
                
            except Exception as e:
                logger.error(f"📡 [WEBSOCKET] Unexpected error for connection {connection_id}: {e}")
                # Send error message to client if connection is still alive
                try:
                    error_message = {
                        "type": "error",
                        "error": "Server error occurred",
                        "timestamp": datetime.now().isoformat(),
                        "connection_id": connection_id
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(error_message, default=str),
                        websocket
                    )
                except:
                    pass  # Connection likely already closed
                break
                
    except WebSocketDisconnect:
        logger.info(f"📡 [WEBSOCKET] Client {connection_id} disconnected during setup for campaign: {campaign_id}")
    except Exception as e:
        logger.error(f"📡 [WEBSOCKET] Failed to establish connection {connection_id} for campaign {campaign_id}: {e}")
    finally:
        websocket_manager.disconnect(websocket, campaign_id)
        logger.info(f"📡 [WEBSOCKET] Connection {connection_id} cleanup completed for campaign: {campaign_id}")