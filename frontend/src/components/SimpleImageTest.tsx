import React, { useState } from 'react';

const SimpleImageTest: React.FC = () => {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const generateImage = async () => {
    setLoading(true);
    setError('');
    
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Simple test - Response:', data);
      
      if (data.images && data.images.length > 0) {
        const url = data.images[0].url;
        console.log('Simple test - Image URL:', url);
        console.log('Simple test - URL type:', typeof url);
        console.log('Simple test - URL length:', url.length);
        setImageUrl(url);
      } else {
        throw new Error('No images received');
      }
    } catch (err) {
      console.error('Simple test - Error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h2>🧪 Simple Image Test</h2>
      
      <button 
        onClick={generateImage}
        disabled={loading}
        style={{
          padding: '10px 20px',
          backgroundColor: '#4F46E5',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: loading ? 'not-allowed' : 'pointer',
          marginBottom: '20px'
        }}
      >
        {loading ? 'Generando...' : 'Generar Imagen'}
      </button>

      {error && (
        <div style={{ color: 'red', marginBottom: '20px' }}>
          Error: {error}
        </div>
      )}

      {imageUrl && (
        <div>
          <h3>Imagen generada:</h3>
          <div style={{ border: '2px solid blue', padding: '10px', marginBottom: '10px' }}>
            <img
              src={imageUrl}
              alt="Generated image"
              style={{ width: '300px', height: '200px', border: '1px solid #ccc' }}
              onLoad={() => console.log('Simple test - Image loaded successfully')}
              onError={(e) => {
                console.error('Simple test - Image failed to load');
                console.error('Simple test - Failed URL:', imageUrl);
                console.error('Simple test - Error event:', e);
              }}
            />
          </div>
          <div style={{ fontSize: '12px', color: '#666', wordBreak: 'break-all' }}>
            <strong>URL:</strong> {imageUrl.substring(0, 100)}...
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleImageTest; 