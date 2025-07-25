#!/usr/bin/env python3
"""
Script de prueba para crear un solo blog y verificar que la API funciona
"""

import requests
import json

# URL de tu API
API_BASE_URL = "https://credilinq-blog-production.up.railway.app"

# Contexto de la empresa
COMPANY_CONTEXT = """Credilinq.ai is a fintech leader in embedded lending and B2B credit solutions across Southeast Asia. We help businesses access funding through embedded financial products and innovative credit infrastructure."""

def test_create_blog():
    """Probar crear un blog"""
    url = f"{API_BASE_URL}/blogs"
    
    payload = {
        "title": "What Is Working Capital—and Why It's Critical for Marketplace Sellers",
        "company_context": COMPANY_CONTEXT,
        "content_type": "blog"
    }
    
    print("🧪 Probando creación de blog...")
    print(f"📡 URL: {url}")
    print(f"📝 Título: {payload['title']}")
    print("-" * 60)
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print("✅ ¡Blog creado exitosamente!")
        print(f"🆔 ID: {result.get('id')}")
        print(f"📄 Título: {result.get('title')}")
        print(f"📊 Estado: {result.get('status')}")
        print(f"🕐 Creado: {result.get('created_at')}")
        print("\n🎉 La API funciona correctamente. ¡Puedes ejecutar el script completo!")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        print("🔧 Verifica que el backend esté funcionando correctamente.")
        return None

if __name__ == "__main__":
    test_create_blog() 