#!/usr/bin/env python3
"""
Script para corregir la configuración de Supabase.
"""

import re
from pathlib import Path

def fix_supabase_config():
    """Corregir la configuración de Supabase."""
    print("🔧 Corrigiendo configuración de Supabase...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ No se encontró el archivo .env")
        return False
    
    # Leer el archivo actual
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Extraer información de la URL actual
    current_supabase_url = re.search(r'SUPABASE_URL="([^"]*)"', content)
    if current_supabase_url:
        current_url = current_supabase_url.group(1)
        
        # Si la URL contiene la información de la base de datos, extraer el project ID
        if "supabase.co" in current_url and "db." in current_url:
            # Extraer project ID de la URL de la base de datos
            project_id = current_url.split("db.")[1].split(".")[0]
            
            # Corregir las variables
            # 1. Corregir SUPABASE_URL
            correct_supabase_url = f"https://{project_id}.supabase.co"
            content = re.sub(
                r'SUPABASE_URL="[^"]*"',
                f'SUPABASE_URL="{correct_supabase_url}"',
                content
            )
            
            # 2. Corregir DATABASE_URL
            correct_database_url = f"postgresql://postgres:credilinqblogs@db.{project_id}.supabase.co:5432/postgres"
            content = re.sub(
                r'DATABASE_URL="[^"]*"',
                f'DATABASE_URL="{correct_database_url}"',
                content
            )
            
            print(f"✅ Project ID detectado: {project_id}")
            print(f"✅ SUPABASE_URL corregido: {correct_supabase_url}")
            print(f"✅ DATABASE_URL corregido: {correct_database_url}")
            
            # Guardar el archivo corregido
            with open(env_path, 'w') as f:
                f.write(content)
            
            print("✅ Archivo .env corregido exitosamente")
            return True
        else:
            print("❌ No se pudo detectar el project ID de Supabase")
            return False
    else:
        print("❌ No se encontró SUPABASE_URL en el archivo .env")
        return False

if __name__ == "__main__":
    success = fix_supabase_config()
    if success:
        print("\n🎉 Configuración corregida. Ahora puedes probar la conexión:")
        print("python3 test_supabase_connection.py")
    else:
        print("\n❌ No se pudo corregir la configuración") 