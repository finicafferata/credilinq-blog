#!/usr/bin/env python3

import os
from dotenv import load_dotenv

def check_environment():
    """Verifica que todas las variables de entorno necesarias estén configuradas"""
    print("🔍 Verificando configuración de variables de entorno...\n")
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Variables requeridas
    required_vars = {
        'OPENAI_API_KEY': 'Clave de OpenAI para embeddings y LLM',
        'SUPABASE_URL': 'URL del proyecto Supabase',
        'SUPABASE_KEY': 'Clave anónima de Supabase',
        'SUPABASE_DB_URL': 'URL de conexión directa a la base de datos',
    }
    
    # Variables opcionales
    optional_vars = {
        'GOOGLE_API_KEY': 'Clave de Google AI (opcional)',
        'SUPABASE_STORAGE_BUCKET': 'Bucket de storage (por defecto: documents)',
        'DATABASE_URL': 'URL de Prisma con pooler',
        'DATABASE_URL_DIRECT': 'URL directa de Prisma'
    }
    
    missing_required = []
    
    print("📋 Variables requeridas:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mostrar solo los primeros caracteres por seguridad
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:10]}..." if len(value) > 10 else value
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: No encontrada - {description}")
            missing_required.append(var)
    
    print("\n📋 Variables opcionales:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if 'KEY' in var or 'PASSWORD' in var:
                display_value = f"{value[:10]}..." if len(value) > 10 else value
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ⚠️  {var}: No encontrada - {description}")
    
    print("\n" + "="*50)
    
    if missing_required:
        print("❌ Configuración incompleta")
        print(f"Faltan {len(missing_required)} variables requeridas: {', '.join(missing_required)}")
        print("\n📝 Para solucionarlo:")
        print("1. Crea un archivo .env en la raíz del proyecto")
        print("2. Copia el contenido de env_template.txt")
        print("3. Reemplaza los valores de ejemplo con tus credenciales reales")
        print("4. Ejecuta este script nuevamente")
        return False
    else:
        print("✅ Todas las variables requeridas están configuradas")
        
        # Intentar conexión a Supabase si las credenciales están disponibles
        if os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_KEY'):
            print("\n🔌 Probando conexión a Supabase...")
            try:
                from supabase import create_client
                supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
                
                # Intentar una consulta simple
                response = supabase.table('blog_posts').select('count').execute()
                print("  ✅ Conexión a Supabase exitosa")
                
            except ImportError:
                print("  ⚠️  Librería supabase no instalada. Ejecuta: pip install supabase")
            except Exception as e:
                print(f"  ❌ Error al conectar con Supabase: {str(e)}")
                if "permission denied" in str(e).lower():
                    print("     💡 Esto podría indicar un problema de permisos RLS")
                    print("     💡 Considera usar la clave de servicio en lugar de la anónima")
        
        return True

if __name__ == "__main__":
    print("🔧 Verificador de Variables de Entorno - Credilinq Agent\n")
    
    # Verificar si existe el archivo .env
    if not os.path.exists('.env'):
        print("❌ No se encontró el archivo .env")
        print("📝 Necesitas crear un archivo .env con tus credenciales")
        print("   Usa env_template.txt como referencia")
        print("\n" + "="*50)
        exit(1)
    
    success = check_environment()
    
    if success:
        print("\n🚀 Everything ready to run the backend!")
        print("   You can run: python main.py")
    else:
        print("\n🔧 Complete the configuration before continuing")
    
    print("\n" + "="*50) 