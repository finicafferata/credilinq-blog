#!/usr/bin/env python3
"""
Test script to verify the image URL format
"""

import requests
import base64
import json

def test_image_url():
    """Test the image URL format from backend"""
    print("🔧 Probando formato de URL de imagen...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/images/generate",
            json={
                "blog_id": "b35cea44-b48a-40df-b868-0860d31e7996",
                "style": "professional",
                "count": 1
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            image_url = data['images'][0]['url']
            
            print(f"✅ URL recibida: {image_url[:100]}...")
            print(f"📏 Longitud de URL: {len(image_url)}")
            print(f"🔗 Comienza con 'data:': {image_url.startswith('data:')}")
            
            # Test if the base64 part is valid
            if image_url.startswith('data:image/svg+xml;base64,'):
                base64_part = image_url.split(',')[1]
                try:
                    decoded = base64.b64decode(base64_part)
                    print(f"✅ Base64 válido, longitud decodificada: {len(decoded)}")
                    print(f"📄 Contenido SVG: {decoded[:100]}...")
                except Exception as e:
                    print(f"❌ Error decodificando base64: {e}")
            else:
                print("❌ URL no tiene formato data:image/svg+xml;base64,")
                
            return image_url
        else:
            print(f"❌ Error en backend: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def create_test_svg():
    """Create a simple test SVG"""
    svg_content = '''<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
        <rect width="800" height="600" fill="#4F46E5"/>
        <text x="400" y="300" font-family="Arial, sans-serif" font-size="24" fill="white" text-anchor="middle">
            Test Image
        </text>
    </svg>'''
    
    svg_encoded = base64.b64encode(svg_content.encode()).decode()
    data_url = f"data:image/svg+xml;base64,{svg_encoded}"
    
    print(f"🔧 URL de prueba creada: {data_url[:100]}...")
    return data_url

if __name__ == "__main__":
    print("🧪 PRUEBA DE FORMATO DE URL")
    print("=" * 40)
    
    # Test backend URL
    backend_url = test_image_url()
    
    print("\n" + "=" * 40)
    
    # Create test URL
    test_url = create_test_svg()
    
    print("\n" + "=" * 40)
    print("📊 COMPARACIÓN:")
    if backend_url and test_url:
        print(f"Backend URL length: {len(backend_url)}")
        print(f"Test URL length: {len(test_url)}")
        print(f"Both start with 'data:': {backend_url.startswith('data:') and test_url.startswith('data:')}") 