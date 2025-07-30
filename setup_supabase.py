#!/usr/bin/env python3
"""
Script para configurar Supabase paso a paso.
"""

import os
import sys
import re
from pathlib import Path

def print_banner():
    """Imprimir banner del script."""
    print("=" * 60)
    print("🚀 Configuración de Supabase para CrediLinQ Agent")
    print("=" * 60)

def check_env_file():
    """Verificar si existe el archivo .env."""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ Archivo .env encontrado")
        return True
    else:
        print("❌ Archivo .env no encontrado")
        return False

def get_supabase_credentials():
    """Obtener credenciales de Supabase del usuario."""
    print("\n📋 Configuración de Supabase")
    print("-" * 40)
    
    print("Para obtener estas credenciales:")
    print("1. Ve a https://supabase.com")
    print("2. Crea un nuevo proyecto o selecciona uno existente")
    print("3. Ve a Settings > API")
    print("4. Copia la URL y la anon key")
    print()
    
    supabase_url = input("🔗 Supabase URL (ej: https://abc123.supabase.co): ").strip()
    supabase_key = input("🔑 Supabase Anon Key: ").strip()
    
    # Validar formato de URL
    if not supabase_url.startswith("https://"):
        print("❌ Error: La URL debe comenzar con https://")
        return None, None
    
    if not supabase_url.endswith(".supabase.co"):
        print("❌ Error: La URL debe terminar con .supabase.co")
        return None, None
    
    return supabase_url, supabase_key

def update_env_file(supabase_url, supabase_key):
    """Actualizar el archivo .env con las credenciales de Supabase."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ Error: No se encontró el archivo .env")
        return False
    
    # Leer el archivo actual
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Actualizar las variables de Supabase
    content = re.sub(
        r'SUPABASE_URL="[^"]*"',
        f'SUPABASE_URL="{supabase_url}"',
        content
    )
    
    content = re.sub(
        r'SUPABASE_KEY="[^"]*"',
        f'SUPABASE_KEY="{supabase_key}"',
        content
    )
    
    # Actualizar DATABASE_URL con la URL de Supabase
    database_url = f"postgresql://postgres:[password]@db.{supabase_url.split('//')[1].split('.')[0]}.supabase.co:5432/postgres"
    content = re.sub(
        r'DATABASE_URL="[^"]*"',
        f'DATABASE_URL="{database_url}"',
        content
    )
    
    # Guardar el archivo actualizado
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Archivo .env actualizado")
    return True

def test_connection():
    """Probar la conexión después de la configuración."""
    print("\n🧪 Probando conexión...")
    
    try:
        # Importar y ejecutar el script de prueba
        from test_supabase_connection import main as test_main
        return test_main()
    except Exception as e:
        print(f"❌ Error al probar la conexión: {e}")
        return False

def main():
    """Función principal."""
    print_banner()
    
    # Verificar archivo .env
    if not check_env_file():
        print("💡 Crea un archivo .env basado en env.example")
        return False
    
    # Obtener credenciales
    supabase_url, supabase_key = get_supabase_credentials()
    
    if not supabase_url or not supabase_key:
        print("❌ Credenciales inválidas")
        return False
    
    # Actualizar archivo .env
    if not update_env_file(supabase_url, supabase_key):
        return False
    
    print("\n✅ Configuración completada!")
    print("\n📝 Próximos pasos:")
    print("1. Configura la contraseña en DATABASE_URL")
    print("2. Ejecuta: python3 test_supabase_connection.py")
    print("3. Si hay errores, verifica las credenciales en Supabase")
    
    # Preguntar si quiere probar la conexión
    test_now = input("\n¿Quieres probar la conexión ahora? (y/n): ").strip().lower()
    
    if test_now == 'y':
        return test_connection()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 