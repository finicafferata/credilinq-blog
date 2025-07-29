#!/usr/bin/env python3
"""
Script de debug para investigar el error 500 en la creación de blogs
"""

import requests
import json

# URL de tu API
API_BASE_URL = "https://credilinq-blog-production.up.railway.app"

def debug_api():
    """Debug detallado de la API"""
    
    # 1. Verificar que el backend esté funcionando
    print("🔍 1. Verificando conexión con el backend...")
    try:
        response = requests.get(f"{API_BASE_URL}/blogs", timeout=10)
        print(f"✅ GET /blogs: {response.status_code}")
        existing_blogs = response.json()
        print(f"📊 Blogs existentes: {len(existing_blogs)}")
    except Exception as e:
        print(f"❌ Error en GET /blogs: {e}")
        return
    
    # 2. Probar creación con datos mínimos
    print("\n🔍 2. Probando creación de blog...")
    url = f"{API_BASE_URL}/blogs"
    
    payload = {
        "title": "Test Blog Title",
        "company_context": "Test company context",
        "content_type": "blog"
    }
    
    print(f"📡 URL: {url}")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)  # Más tiempo por si tarda el AI
        
        print(f"🔢 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ¡Éxito! Blog creado: {result}")
        else:
            print(f"❌ Error {response.status_code}")
            print(f"📄 Response text: {response.text[:500]}...")
            
            # Intentar parsear como JSON si es posible
            try:
                error_json = response.json()
                print(f"📄 Error JSON: {json.dumps(error_json, indent=2)}")
            except:
                print("📄 No se pudo parsear como JSON")
                
    except requests.exceptions.Timeout:
        print("⏰ Timeout - El agente de IA puede estar tardando mucho")
    except Exception as e:
        print(f"❌ Error en request: {e}")

if __name__ == "__main__":
    debug_api() 