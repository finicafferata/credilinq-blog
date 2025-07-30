#!/usr/bin/env python3
"""
Demo script for Image Agent
Tests the image generation functionality
"""

import asyncio
import json
import requests
from typing import Dict, Any

def test_image_generation():
    """Test image generation with sample blog content"""
    
    print("🚀 Probando generación de imágenes con contenido nuevo...")
    
    # Sample blog content
    blog_title = "Cómo Optimizar tu Estrategia de Marketing Digital en 2024"
    blog_content = """
    En el dinámico mundo del marketing digital, mantenerse actualizado es crucial para el éxito empresarial. 
    Este artículo te guiará a través de las estrategias más efectivas para optimizar tu presencia digital.
    
    ## 1. Análisis de Datos y Métricas
    
    El primer paso para optimizar tu estrategia es implementar un sistema robusto de análisis de datos. 
    Utiliza herramientas como Google Analytics, Facebook Insights y LinkedIn Analytics para obtener 
    información valiosa sobre el comportamiento de tu audiencia.
    
    ## 2. Personalización de Contenido
    
    La personalización es clave en el marketing moderno. Segmenta tu audiencia y crea contenido 
    específico para cada grupo demográfico. Esto aumentará significativamente tus tasas de engagement.
    
    ## 3. Automatización de Marketing
    
    Implementa herramientas de automatización para optimizar tus procesos de marketing. 
    Esto te permitirá escalar tus esfuerzos de manera eficiente.
    """
    
    # Test data
    test_data = {
        "content": blog_content,
        "blog_title": blog_title,
        "outline": [
            "Análisis de Datos y Métricas",
            "Personalización de Contenido", 
            "Automatización de Marketing"
        ],
        "style": "professional",
        "count": 3
    }
    
    print("🚀 Iniciando prueba del Agente de Imágenes...")
    print(f"📝 Título del blog: {blog_title}")
    print(f"🎨 Estilo seleccionado: {test_data['style']}")
    print(f"📊 Cantidad de imágenes: {test_data['count']}")
    print("-" * 50)
    
    try:
        # Test direct API call
        print("📡 Probando generación de imágenes...")
        response = requests.post(
            "http://localhost:8000/api/images/generate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Imágenes generadas exitosamente!")
            print(f"📸 Total de imágenes: {len(result['images'])}")
            
            for i, image in enumerate(result['images'], 1):
                print(f"\n🖼️  Imagen {i}:")
                print(f"   ID: {image['id']}")
                print(f"   URL: {image['url']}")
                print(f"   Estilo: {image['style']}")
                print(f"   Tamaño: {image['size']}")
                print(f"   Prompt: {image['prompt'][:100]}...")
                
        else:
            print(f"❌ Error en la generación: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error de conexión. Asegúrate de que el servidor esté ejecutándose en http://localhost:8000")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

def test_image_regeneration():
    """Test image regeneration functionality"""
    
    print("\n" + "=" * 50)
    print("🔄 Probando regeneración de imágenes...")
    
    test_data = {
        "style": "creative",
        "blog_title": "Innovación en Tecnología 2024",
        "content": "La tecnología está evolucionando rápidamente..."
    }
    
    try:
        # Test regeneration
        response = requests.post(
            "http://localhost:8000/api/images/regenerate/img_1",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Imagen regenerada exitosamente!")
            print(f"🖼️  Nueva imagen:")
            print(f"   ID: {result['id']}")
            print(f"   URL: {result['url']}")
            print(f"   Estilo: {result['style']}")
        else:
            print(f"❌ Error en regeneración: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error en regeneración: {str(e)}")

def test_available_styles():
    """Test getting available styles"""
    
    print("\n" + "=" * 50)
    print("🎨 Probando estilos disponibles...")
    
    try:
        response = requests.get("http://localhost:8000/api/images/styles")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Estilos obtenidos exitosamente!")
            print(f"📋 Total de estilos: {len(result['styles'])}")
            
            for style in result['styles']:
                print(f"   • {style['name']}: {style['description']}")
        else:
            print(f"❌ Error obteniendo estilos: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error obteniendo estilos: {str(e)}")

def test_existing_blogs():
    """Test getting available blogs"""
    
    print("\n" + "=" * 50)
    print("📚 Probando obtención de blogs existentes...")
    
    try:
        response = requests.get("http://localhost:8000/api/images/blogs")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Blogs obtenidos exitosamente!")
            print(f"📋 Total de blogs: {len(result['blogs'])}")
            
            for blog in result['blogs']:
                print(f"   • {blog['title']} (ID: {blog['id']}, Status: {blog['status']})")
            
            # Si hay blogs disponibles, probar generación con el primero
            if result['blogs']:
                first_blog = result['blogs'][0]
                print(f"\n🖼️  Probando generación con blog existente: {first_blog['title']}")
                
                test_data = {
                    "blog_id": first_blog['id'],
                    "style": "professional",
                    "count": 2
                }
                
                response = requests.post(
                    "http://localhost:8000/api/images/generate",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Imágenes generadas exitosamente desde blog existente!")
                    print(f"📸 Total de imágenes: {len(result['images'])}")
                else:
                    print(f"❌ Error generando imágenes desde blog existente: {response.status_code}")
            else:
                print("ℹ️  No hay blogs disponibles para probar")
        else:
            print(f"❌ Error obteniendo blogs: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error obteniendo blogs: {str(e)}")

def main():
    """Main function to run all tests"""
    
    print("🎯 DEMO: Agente de Imágenes - CrediLinQ")
    print("=" * 50)
    
    # Run tests
    test_image_generation()
    test_image_regeneration()
    test_available_styles()
    test_existing_blogs()
    
    print("\n" + "=" * 50)
    print("✅ Demo completado!")
    print("\n💡 Próximos pasos:")
    print("   1. Abre http://localhost:3000/image-agent en tu navegador")
    print("   2. Prueba la interfaz web del agente de imágenes")
    print("   3. Experimenta con diferentes estilos y configuraciones")

if __name__ == "__main__":
    main() 