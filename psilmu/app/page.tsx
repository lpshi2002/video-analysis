
'use client';

import React, { useRef, useState } from 'react';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const requestRef = useRef<number>(0);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<string>('Ready');

  const startStreaming = async () => {
    try {
      setStatus('Starting Camera...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setStatus('Connecting to Server...');
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('Streaming Active');
        setIsStreaming(true);
        sendFrame();
      };

      ws.onclose = () => {
        stopStreaming();
        setStatus('Connection Closed');
      };

      ws.onerror = (err) => {
        console.error(err);
        setStatus('WebSocket Error');
        stopStreaming();
      };

    } catch (err) {
      console.error(err);
      setStatus('Permission Denied / Error');
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }

    if (videoRef.current && videoRef.current.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setStatus('Stopped');
  };

  const sendFrame = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 품질 0.5로 JPEG 변환하여 전송
        const base64Data = canvas.toDataURL('image/jpeg', 0.5);
        wsRef.current.send(base64Data);
      }
    }
    requestRef.current = requestAnimationFrame(sendFrame);
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      minHeight: '100vh', backgroundColor: '#f0f0f0', fontFamily: 'sans-serif'
    }}>
      <h1 style={{ marginBottom: '20px' }}>MediaPipe Face Landmarker Test</h1>
      <div style={{ padding: '8px 16px', background: '#ddd', borderRadius: '4px', marginBottom: '10px' }}>
        Status: <strong>{status}</strong>
      </div>

      <div style={{ border: '2px solid #333', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
        <video ref={videoRef} style={{ display: 'block', width: '640px', maxWidth: '100%' }} muted playsInline />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <button 
        onClick={isStreaming ? stopStreaming : startStreaming}
        style={{
          marginTop: '20px', padding: '12px 24px', fontSize: '16px', cursor: 'pointer',
          backgroundColor: isStreaming ? '#d32f2f' : '#1976d2', color: 'white', border: 'none', borderRadius: '4px'
        }}
      >
        {isStreaming ? 'STOP Analysis' : 'START Analysis'}
      </button>
    </div>
  );
}