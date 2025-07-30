import React, { useState } from 'react';

const DebugImageTest: React.FC = () => {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [debugInfo, setDebugInfo] = useState<string>('');

  const testWithHardcodedUrl = () => {
    // Test with a hardcoded data URL first
    const testUrl = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzRGNDZFNSIvPgogIDx0ZXh0IHg9IjE1MCIgeT0iMTAwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5UZXN0IEltYWdlPC90ZXh0Pgo8L3N2Zz4K";
    console.log('Debug - Setting hardcoded URL:', testUrl);
    setImageUrl(testUrl);
    setDebugInfo('Hardcoded URL set');
  };

  const testWithBackendUrl = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/images/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          blog_id: 'b35cea44-b48a-40df-b868-0860d31e7996',
          style: 'professional',
          count: 1
        })
      });

      const data = await response.json();
      console.log('Debug - Backend response:', data);
      
      if (data.images && data.images.length > 0) {
        const url = data.images[0].url;
        console.log('Debug - Setting backend URL:', url);
        setImageUrl(url);
        setDebugInfo(`Backend URL set (length: ${url.length})`);
      }
    } catch (error) {
      console.error('Debug - Error:', error);
      setDebugInfo(`Error: ${error}`);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h2>🐛 Debug Image Test</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={testWithHardcodedUrl}
          style={{
            padding: '10px 20px',
            backgroundColor: '#10B981',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginRight: '10px'
          }}
        >
          Test Hardcoded URL
        </button>
        
        <button 
          onClick={testWithBackendUrl}
          style={{
            padding: '10px 20px',
            backgroundColor: '#3B82F6',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Test Backend URL
        </button>
      </div>

      {debugInfo && (
        <div style={{ 
          padding: '10px', 
          backgroundColor: '#F3F4F6', 
          borderRadius: '4px',
          marginBottom: '20px',
          fontSize: '14px'
        }}>
          <strong>Debug Info:</strong> {debugInfo}
        </div>
      )}

      {imageUrl && (
        <div>
          <h3>Imagen de prueba:</h3>
          <div style={{ 
            border: '3px solid red', 
            padding: '15px', 
            marginBottom: '10px',
            backgroundColor: '#FEF2F2'
          }}>
            <img
              src={imageUrl}
              alt="Debug test image"
              style={{ 
                width: '300px', 
                height: '200px', 
                border: '2px solid blue',
                display: 'block'
              }}
              onLoad={() => {
                console.log('Debug - Image loaded successfully!');
                setDebugInfo(prev => prev + ' | Image loaded successfully');
              }}
              onError={(e) => {
                console.error('Debug - Image failed to load!');
                console.error('Debug - Failed URL:', imageUrl);
                setDebugInfo(prev => prev + ' | Image failed to load');
              }}
            />
          </div>
          
          <div style={{ 
            fontSize: '12px', 
            color: '#666', 
            wordBreak: 'break-all',
            backgroundColor: '#F9FAFB',
            padding: '10px',
            borderRadius: '4px'
          }}>
            <strong>Current URL:</strong><br/>
            {imageUrl.substring(0, 100)}...
          </div>
        </div>
      )}
    </div>
  );
};

export default DebugImageTest; 