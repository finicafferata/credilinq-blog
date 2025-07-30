#!/usr/bin/env python3
"""
Test script to verify frontend-backend integration for image generation
"""

import requests
import json
import time

def test_backend_images():
    """Test backend image generation"""
    print("🔧 Probando generación de imágenes en el backend...")
    
    # Test data
    test_data = {
        "blog_id": "b35cea44-b48a-40df-b868-0860d31e7996",
        "style": "professional",
        "count": 2
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/images/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend funciona correctamente")
            print(f"📸 Imágenes generadas: {len(data['images'])}")
            
            for i, image in enumerate(data['images'], 1):
                print(f"   Imagen {i}: {image['url'][:50]}...")
                # Test if the image URL is accessible
                if image['url'].startswith('data:'):
                    print(f"   ✅ Data URL válida")
                else:
                    print(f"   ⚠️  URL externa: {image['url']}")
            
            return data['images']
        else:
            print(f"❌ Error en backend: {response.status_code}")
            print(f"Respuesta: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error probando backend: {str(e)}")
        return None

def test_frontend_access():
    """Test if frontend is accessible"""
    print("\n🌐 Probando acceso al frontend...")
    
    try:
        response = requests.get("http://localhost:5173")
        if response.status_code == 200:
            print("✅ Frontend accesible")
            return True
        else:
            print(f"❌ Frontend no accesible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accediendo al frontend: {str(e)}")
        return False

def test_blogs_endpoint():
    """Test blogs endpoint"""
    print("\n📝 Probando endpoint de blogs...")
    
    try:
        response = requests.get("http://localhost:8000/api/images/blogs")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blogs disponibles: {len(data['blogs'])}")
            return True
        else:
            print(f"❌ Error en endpoint de blogs: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error probando blogs: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 PRUEBA COMPLETA: Frontend-Backend Integration")
    print("=" * 50)
    
    # Test backend
    images = test_backend_images()
    
    # Test frontend
    frontend_ok = test_frontend_access()
    
    # Test blogs
    blogs_ok = test_blogs_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   Backend: {'✅ OK' if images else '❌ FALLO'}")
    print(f"   Frontend: {'✅ OK' if frontend_ok else '❌ FALLO'}")
    print(f"   Blogs: {'✅ OK' if blogs_ok else '❌ FALLO'}")
    
    if images and frontend_ok and blogs_ok:
        print("\n🎉 ¡Todo funciona correctamente!")
        print("\n💡 Para probar en el navegador:")
        print("   1. Ve a http://localhost:5173")
        print("   2. Navega a la página de imágenes")
        print("   3. Selecciona 'Blog existente'")
        print("   4. Elige un blog y genera imágenes")
        print("   5. Abre la consola del navegador (F12) para ver los logs")
    else:
        print("\n⚠️  Hay problemas que necesitan atención")
        
        if not images:
            print("   - Backend no está generando imágenes correctamente")
        if not frontend_ok:
            print("   - Frontend no está accesible")
        if not blogs_ok:
            print("   - Endpoint de blogs no funciona")

if __name__ == "__main__":
    main() 