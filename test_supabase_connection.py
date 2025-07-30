#!/usr/bin/env python3
"""
Script para probar la conexión con Supabase.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Cargar variables de entorno."""
    load_dotenv()
    
    # Variables requeridas
    required_vars = [
        'DATABASE_URL',
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Variables de entorno faltantes: {', '.join(missing_vars)}")
        logger.info("💡 Asegúrate de tener un archivo .env con las variables necesarias")
        return False
    
    return True

def test_database_connection():
    """Probar conexión directa a la base de datos."""
    try:
        database_url = os.getenv('DATABASE_URL')
        logger.info("🔗 Probando conexión a la base de datos...")
        
        # Conectar a la base de datos
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Probar consulta simple
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✅ Conexión exitosa a PostgreSQL: {version[0]}")
        
        # Probar consulta de información de la base de datos
        cursor.execute("SELECT current_database(), current_user, inet_server_addr();")
        db_info = cursor.fetchone()
        logger.info(f"📊 Base de datos: {db_info[0]}")
        logger.info(f"👤 Usuario: {db_info[1]}")
        logger.info(f"🌐 Servidor: {db_info[2]}")
        
        # Verificar tablas existentes
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            logger.info(f"📋 Tablas encontradas ({len(tables)}):")
            for table in tables:
                logger.info(f"   - {table[0]}")
        else:
            logger.warning("⚠️  No se encontraron tablas en la base de datos")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        logger.error(f"❌ Error de conexión a la base de datos: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

def test_supabase_client():
    """Probar cliente de Supabase."""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        logger.info("🔗 Probando cliente de Supabase...")
        
        # Crear cliente de Supabase
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Probar autenticación
        auth_response = supabase.auth.get_user()
        logger.info("✅ Cliente de Supabase configurado correctamente")
        
        # Probar acceso a la base de datos
        try:
            # Intentar una consulta simple
            response = supabase.table('blogs').select('*').limit(1).execute()
            logger.info("✅ Acceso a la base de datos de Supabase exitoso")
            logger.info(f"📊 Datos de ejemplo: {len(response.data)} registros encontrados")
        except Exception as e:
            logger.warning(f"⚠️  No se pudo acceder a la tabla 'blogs': {e}")
            logger.info("💡 Esto es normal si la tabla no existe aún")
        
        return True
        
    except ImportError:
        logger.error("❌ Cliente de Supabase no instalado. Instala con: pip install supabase")
        return False
    except Exception as e:
        logger.error(f"❌ Error con el cliente de Supabase: {e}")
        return False

def test_environment_variables():
    """Mostrar información de las variables de entorno."""
    logger.info("🔧 Variables de entorno configuradas:")
    
    # Variables sensibles (mostrar solo parte)
    sensitive_vars = ['DATABASE_URL', 'SUPABASE_KEY', 'OPENAI_API_KEY']
    
    for var in ['DATABASE_URL', 'SUPABASE_URL', 'SUPABASE_KEY', 'OPENAI_API_KEY']:
        value = os.getenv(var)
        if value:
            if var in sensitive_vars:
                # Mostrar solo los primeros caracteres
                display_value = value[:20] + "..." if len(value) > 20 else value
                logger.info(f"   {var}: {display_value}")
            else:
                logger.info(f"   {var}: {value}")
        else:
            logger.warning(f"   {var}: ❌ No configurada")

def main():
    """Función principal."""
    logger.info("🚀 Iniciando prueba de conexión con Supabase...")
    logger.info("=" * 50)
    
    # Cargar variables de entorno
    if not load_environment():
        sys.exit(1)
    
    # Mostrar variables de entorno
    test_environment_variables()
    logger.info("=" * 50)
    
    # Probar conexión a la base de datos
    db_success = test_database_connection()
    logger.info("=" * 50)
    
    # Probar cliente de Supabase
    supabase_success = test_supabase_client()
    logger.info("=" * 50)
    
    # Resumen
    if db_success and supabase_success:
        logger.info("🎉 ¡Todas las pruebas pasaron! Supabase está configurado correctamente.")
        return True
    elif db_success:
        logger.info("⚠️  Conexión a la base de datos OK, pero hay problemas con el cliente de Supabase.")
        return False
    else:
        logger.error("❌ Hay problemas con la configuración de Supabase.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 