// useWebSocket.js
import { useCallback } from 'react';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import { webSocketAtom } from '../state/webSocketAtom';
import { notificationAtom } from '../state/notificationAtom';

export function useWebSocket() {
  const ws = useRecoilValue(webSocketAtom);
  const setNotification = useSetRecoilState(notificationAtom);

  // A helper function for sending strategy requests.
  const sendStrategyRequest = useCallback((strategyRequest) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.error('No active WebSocket connection', ws?.readyState);
      setNotification({
        message: 'Connection lost. Please try again.',
        type: 'error'
      });
      return;
    }
    const payload = {
      command: 'execute_strategy',
      strategy: strategyRequest,
    };
    console.log('Sending strategy request:', payload);
    try {
      ws.send(JSON.stringify(payload));
    } catch (error) {
      console.error('Error sending strategy request:', error);
      setNotification({
        message: 'Error sending request. Please try again.',
        type: 'error'
      });
    }
  }, [ws, setNotification]);

  return { 
    sendStrategyRequest,
    isConnected: ws?.readyState === WebSocket.OPEN 
  };
}