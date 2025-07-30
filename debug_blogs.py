#!/usr/bin/env python3
"""
Debug script to check blogs in database
"""

import requests
import json
from typing import Dict, Any

def check_blogs_api():
    """Check if blogs are available via API"""
    
    print("🔍 Verificando blogs disponibles...")
    print("=" * 50)
    
    try:
        # Check main blogs endpoint
        print("1. Verificando endpoint principal de blogs...")
        response = requests.get("http://localhost:8000/api/blogs")
        
        if response.status_code == 200:
            blogs = response.json()
            print(f"✅ Encontrados {len(blogs)} blogs en el endpoint principal")
            for blog in blogs:
                print(f"   • {blog['title']} (ID: {blog['id']}, Status: {blog['status']})")
        else:
            print(f"❌ Error en endpoint principal: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except Exception as e:
        print(f"❌ Error verificando endpoint principal: {str(e)}")
    
    print("\n" + "=" * 50)
    
    try:
        # Check images blogs endpoint
        print("2. Verificando endpoint de blogs para imágenes...")
        response = requests.get("http://localhost:8000/api/images/blogs")
        
        if response.status_code == 200:
            data = response.json()
            blogs = data.get('blogs', [])
            print(f"✅ Encontrados {len(blogs)} blogs en el endpoint de imágenes")
            for blog in blogs:
                print(f"   • {blog['title']} (ID: {blog['id']}, Status: {blog['status']})")
        else:
            print(f"❌ Error en endpoint de imágenes: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except Exception as e:
        print(f"❌ Error verificando endpoint de imágenes: {str(e)}")

def test_image_generation_with_existing_blog():
    """Test image generation with an existing blog"""
    
    print("\n" + "=" * 50)
    print("🖼️  Probando generación de imágenes con blog existente...")
    
    try:
        # First, get available blogs
        response = requests.get("http://localhost:8000/api/images/blogs")
        
        if response.status_code == 200:
            data = response.json()
            blogs = data.get('blogs', [])
            
            if blogs:
                # Use the first blog
                first_blog = blogs[0]
                print(f"📝 Usando blog: {first_blog['title']}")
                
                # Test image generation
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
                    print("✅ Generación exitosa!")
                    print(f"📸 Imágenes generadas: {len(result['images'])}")
                    
                    for i, image in enumerate(result['images'], 1):
                        print(f"   Imagen {i}: {image['prompt'][:50]}...")
                else:
                    print(f"❌ Error en generación: {response.status_code}")
                    print(f"Respuesta: {response.text}")
            else:
                print("ℹ️  No hay blogs disponibles para probar")
        else:
            print(f"❌ Error obteniendo blogs: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error en prueba: {str(e)}")

def check_database_directly():
    """Check database directly"""
    
    print("\n" + "=" * 50)
    print("🗄️  Verificando base de datos directamente...")
    
    try:
        from src.config.database import db_config
        
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            
            # Check if BlogPost table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'BlogPost'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                print("✅ Tabla BlogPost existe")
                
                # Count blogs
                cur.execute("SELECT COUNT(*) FROM \"BlogPost\"")
                total_count = cur.fetchone()[0]
                print(f"📊 Total de blogs en BD: {total_count}")
                
                # Get non-deleted blogs
                cur.execute("""
                    SELECT id, title, status, "createdAt"
                    FROM "BlogPost" 
                    WHERE status != 'deleted' 
                    ORDER BY "createdAt" DESC
                """)
                rows = cur.fetchall()
                
                print(f"📝 Blogs no eliminados: {len(rows)}")
                for row in rows:
                    print(f"   • {row[1]} (ID: {row[0]}, Status: {row[2]})")
            else:
                print("❌ Tabla BlogPost no existe")
                
    except Exception as e:
        print(f"❌ Error verificando BD: {str(e)}")

def main():
    """Main function"""
    
    print("🔧 DIAGNÓSTICO: Verificación de Blogs")
    print("=" * 50)
    
    # Check database directly
    check_database_directly()
    
    # Check API endpoints
    check_blogs_api()
    
    # Test image generation
    test_image_generation_with_existing_blog()
    
    print("\n" + "=" * 50)
    print("✅ Diagnóstico completado!")
    print("\n💡 Si no ves blogs, verifica:")
    print("   1. Que el servidor esté ejecutándose")
    print("   2. Que hayas creado blogs anteriormente")
    print("   3. Que la base de datos esté configurada correctamente")

if __name__ == "__main__":
    main() 