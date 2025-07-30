#!/usr/bin/env python3
"""
Script para verificar el estado actual de la configuración de Supabase.
"""

import os
from dotenv import load_dotenv

def main():
    """Verificar configuración de Supabase."""
    print("🔍 Verificando configuración de Supabase...")
    print("=" * 50)
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Variables a verificar
    variables = {
        'DATABASE_URL': os.getenv('DATABASE_URL'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    print("📋 Estado de las variables de entorno:")
    print("-" * 40)
    
    all_configured = True
    
    for var, value in variables.items():
        if value and value != "your_supabase_url" and value != "your_supabase_anon_key" and value != "your_openai_api_key":
            print(f"✅ {var}: Configurada")
        else:
            print(f"❌ {var}: No configurada o con valor por defecto")
            all_configured = False
    
    print("\n" + "=" * 50)
    
    if all_configured:
        print("🎉 ¡Todas las variables están configuradas!")
        print("💡 Puedes ejecutar: python3 test_supabase_connection.py")
    else:
        print("⚠️  Algunas variables necesitan configuración")
        print("\n📝 Para configurar Supabase:")
        print("1. Ve a https://supabase.com")
        print("2. Crea un nuevo proyecto")
        print("3. Ve a Settings > API")
        print("4. Copia la URL y anon key")
        print("5. Actualiza el archivo .env")
        print("\n💡 O ejecuta: python3 setup_supabase.py")
    
    return all_configured

if __name__ == "__main__":
    main() 