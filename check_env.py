#!/usr/bin/env python3
"""
Script para verificar que las variables de entorno estén configuradas correctamente
"""
import os
from dotenv import load_dotenv

def check_environment():
    print("🔍 Verificando variables de entorno...")
    print("=" * 50)
    
    load_dotenv()
    
    # Variables requeridas
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'SUPABASE_URL': 'Supabase Project URL',
        'SUPABASE_KEY': 'Supabase Anon Key',
        'SUPABASE_DB_URL': 'Supabase Database URL'
    }
    
    # Variables opcionales
    optional_vars = {
        'GOOGLE_API_KEY': 'Google AI API Key',
        'SUPABASE_STORAGE_BUCKET': 'Supabase Storage Bucket'
    }
    
    missing_required = []
    
    print("📋 Variables Requeridas:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * 10}...{value[-4:] if len(value) > 4 else '****'}")
        else:
            print(f"  ❌ {var}: NO CONFIGURADA")
            missing_required.append(var)
    
    print("\n📋 Variables Opcionales:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * 10}...{value[-4:] if len(value) > 4 else '****'}")
        else:
            print(f"  ⚠️  {var}: No configurada (opcional)")
    
    print("\n" + "=" * 50)
    
    if missing_required:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        print(f"Variables faltantes: {', '.join(missing_required)}")
        print("\n📝 Para solucionarlo:")
        print("1. Crea un archivo .env en la raíz del proyecto")
        print("2. Agrega las variables faltantes con sus valores reales")
        return False
    else:
        print("✅ CONFIGURACIÓN COMPLETA")
        print("Todas las variables requeridas están configuradas.")
        return True

if __name__ == "__main__":
    check_environment() 