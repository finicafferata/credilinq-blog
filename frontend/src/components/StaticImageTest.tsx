import React, { useState } from 'react';

const StaticImageTest: React.FC = () => {
  const [testImage, setTestImage] = useState<string>('');

  const testStaticImage = () => {
    // Test with a simple static image
    const staticUrl = "https://httpbin.org/image/png";
    setTestImage(staticUrl);
  };

  const testDataUrl = () => {
    // Test with a simple data URL
    const dataUrl = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzRGNDZFNSIvPgogIDx0ZXh0IHg9IjE1MCIgeT0iMTAwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5UZXN0PC90ZXh0Pgo8L3N2Zz4K";
    setTestImage(dataUrl);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h2>🧪 Static Image Test</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={testStaticImage}
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
          Test Static URL
        </button>
        
        <button 
          onClick={testDataUrl}
          style={{
            padding: '10px 20px',
            backgroundColor: '#3B82F6',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Test Data URL
        </button>
      </div>

      {testImage && (
        <div>
          <h3>Imagen de prueba:</h3>
          <div style={{ 
            border: '3px solid green', 
            padding: '15px', 
            marginBottom: '10px',
            backgroundColor: '#F0FDF4'
          }}>
            <img
              src={testImage}
              alt="Test image"
              style={{ 
                width: '300px', 
                height: '200px', 
                border: '2px solid blue',
                display: 'block'
              }}
              onLoad={() => console.log('Static test - Image loaded successfully!')}
              onError={(e) => {
                console.error('Static test - Image failed to load!');
                console.error('Static test - Failed URL:', testImage);
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
            {testImage.substring(0, 100)}...
          </div>
        </div>
      )}
    </div>
  );
};

export default StaticImageTest; 