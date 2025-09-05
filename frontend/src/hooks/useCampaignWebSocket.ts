import { useEffect, useState, useCallback, useRef } from 'react';

// Enhanced WebSocket message types from backend
export interface CampaignStatusMessage {
  type: 'connection_established' | 'workflow_started' | 'agents_starting' | 'workflow_completed' | 'campaign_completed' | 'echo' | 'heartbeat' | 'pong' | 'error';
  campaign_id: string;
  connection_id?: string;
  agent_type?: string;
  status?: 'running' | 'completed' | 'failed';
  message: string;
  progress?: number;
  timestamp: string;
  server_time?: string;
  uptime_seconds?: number;
  error?: string;
  received_data?: string;
  message_type?: string;
  results?: {
    success?: boolean;
    agents_executed?: string[];
    execution_time?: number;
  };
  content_created?: {
    title: string;
    type: string;
    word_count: number;
    status: string;
  };
  received?: string; // For echo messages
}

export interface UseCampaignWebSocketReturn {
  isConnected: boolean;
  lastMessage: CampaignStatusMessage | null;
  messages: CampaignStatusMessage[];
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  progress: number;
  currentStatus: string;
  connectionId: string | null;
  lastHeartbeat: Date | null;
  sendMessage: (message: string) => void;
  sendPing: () => void;
  clearMessages: () => void;
}

/**
 * Phase 3: React WebSocket hook for real-time campaign status updates
 * 
 * @param campaignId - The campaign ID to listen for updates
 * @param enabled - Whether to connect to WebSocket (default: true)
 * @returns WebSocket connection state and received messages
 */
export const useCampaignWebSocket = (
  campaignId: string | null | undefined, 
  enabled: boolean = true
): UseCampaignWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<CampaignStatusMessage | null>(null);
  const [messages, setMessages] = useState<CampaignStatusMessage[]>([]);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [progress, setProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState('Waiting for campaign updates...');
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [lastHeartbeat, setLastHeartbeat] = useState<Date | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const pingInterval = 30000; // 30 seconds

  const getWebSocketUrl = useCallback((campaignId: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = window.location.hostname === 'localhost' ? '8000' : window.location.port;
    const portSuffix = port ? `:${port}` : '';
    return `${protocol}//${host}${portSuffix}/api/v2/campaigns/ws/campaign/${campaignId}/status`;
  }, []);

  const sendMessage = useCallback((message: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(message);
        return true;
      } catch (error) {
        console.error('🔌 [WEBSOCKET] Failed to send message:', error);
        return false;
      }
    }
    console.warn('🔌 [WEBSOCKET] Cannot send message: WebSocket not connected');
    return false;
  }, []);

  const sendPing = useCallback(() => {
    const pingMessage = JSON.stringify({
      type: 'ping',
      timestamp: new Date().toISOString(),
      campaign_id: campaignId
    });
    return sendMessage(pingMessage);
  }, [sendMessage, campaignId]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setLastMessage(null);
    setProgress(0);
    setCurrentStatus('Waiting for campaign updates...');
    setConnectionId(null);
    setLastHeartbeat(null);
  }, []);

  const processMessage = useCallback((message: CampaignStatusMessage) => {
    // Handle different message types appropriately
    switch (message.type) {
      case 'connection_established':
        setConnectionId(message.connection_id || null);
        setCurrentStatus('Connected to campaign updates');
        setProgress(0);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'heartbeat':
        setLastHeartbeat(new Date());
        // Don't add heartbeat messages to the main message list to avoid clutter
        break;
        
      case 'pong':
        setLastHeartbeat(new Date());
        console.log('🔌 [WEBSOCKET] Received pong response');
        // Don't add pong messages to the main message list
        break;
        
      case 'error':
        console.error('🔌 [WEBSOCKET] Received error from server:', message.error);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'workflow_started':
        setCurrentStatus('Starting content generation workflow...');
        setProgress(message.progress || 0);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'agents_starting':
        setCurrentStatus('AI agents analyzing and generating content...');
        setProgress(message.progress || 25);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'workflow_completed':
        const success = message.results?.success || false;
        setCurrentStatus(success ? 'Content generation completed successfully' : 'Content generation failed');
        setProgress(message.progress || (success ? 75 : 0));
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'campaign_completed':
        setCurrentStatus('Campaign content generated and ready!');
        setProgress(100);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      case 'echo':
        // Echo messages are mainly for testing - add to messages but don't update status
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
        
      default:
        console.warn('🔌 [WEBSOCKET] Unknown message type:', message.type);
        setLastMessage(message);
        setMessages(prev => [...prev, message]);
        break;
    }
  }, []);

  const connect = useCallback(() => {
    if (!campaignId || !enabled) return;

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('connecting');
    setCurrentStatus('Connecting to real-time updates...');
    
    try {
      const wsUrl = getWebSocketUrl(campaignId);
      console.log(`🔌 [WEBSOCKET] Connecting to: ${wsUrl}`);
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log(`🔌 [WEBSOCKET] Connected to campaign: ${campaignId}`);
        setIsConnected(true);
        setConnectionState('connected');
        reconnectAttempts.current = 0;
        
        // Start ping interval to keep connection alive
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        pingIntervalRef.current = setInterval(() => {
          sendPing();
        }, pingInterval);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message: CampaignStatusMessage = JSON.parse(event.data);
          console.log(`📨 [WEBSOCKET] Received message:`, message);
          processMessage(message);
        } catch (error) {
          console.error('🔌 [WEBSOCKET] Error parsing message:', error, 'Raw data:', event.data);
          // Create error message to display to user
          const errorMessage: CampaignStatusMessage = {
            type: 'error',
            campaign_id: campaignId,
            message: 'Failed to parse server message',
            error: error instanceof Error ? error.message : 'Unknown parsing error',
            timestamp: new Date().toISOString()
          };
          processMessage(errorMessage);
        }
      };
      
      wsRef.current.onclose = (event) => {
        console.log(`🔌 [WEBSOCKET] Connection closed for campaign: ${campaignId}`, event.code, event.reason);
        setIsConnected(false);
        setConnectionState('disconnected');
        setCurrentStatus('Connection lost. Attempting to reconnect...');
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        // Auto-reconnect logic with improved backoff
        if (enabled && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          console.log(`🔌 [WEBSOCKET] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (enabled) { // Check if still enabled before reconnecting
              connect();
            }
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setCurrentStatus('Connection failed. Please refresh the page.');
          setConnectionState('error');
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error(`🔌 [WEBSOCKET] Error for campaign: ${campaignId}`, error);
        setConnectionState('error');
        setCurrentStatus('WebSocket connection error occurred');
      };
      
    } catch (error) {
      console.error('🔌 [WEBSOCKET] Failed to create WebSocket connection:', error);
      setConnectionState('error');
      setCurrentStatus('Failed to create WebSocket connection');
    }
  }, [campaignId, enabled, getWebSocketUrl, processMessage, sendPing]);

  const disconnect = useCallback(() => {
    // Clean up all timers and connections
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      console.log(`🔌 [WEBSOCKET] Disconnecting from campaign: ${campaignId}`);
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionState('disconnected');
    setCurrentStatus('Disconnected from campaign updates');
    reconnectAttempts.current = 0;
  }, [campaignId]);

  // Connect when component mounts or campaignId changes
  useEffect(() => {
    if (campaignId && enabled) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [campaignId, enabled, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    lastMessage,
    messages,
    connectionState,
    progress,
    currentStatus,
    connectionId,
    lastHeartbeat,
    sendMessage,
    sendPing,
    clearMessages
  };
};