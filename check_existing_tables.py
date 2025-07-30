#!/usr/bin/env python3
"""
Script para verificar todas las tablas existentes en Supabase.
"""

import psycopg2
import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_existing_tables():
    """Verificar todas las tablas existentes en Supabase."""
    load_dotenv()
    
    try:
        # Conectar a la base de datos
        database_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🔗 Conectado a Supabase")
        
        # Obtener todas las tablas
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        logger.info(f"📋 Tablas encontradas ({len(tables)}):")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        # Para cada tabla, mostrar información adicional
        for table in tables:
            table_name = table[0]
            logger.info(f"\n📊 Información de la tabla '{table_name}':")
            
            # Contar registros
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                count = cursor.fetchone()[0]
                logger.info(f"   📈 Registros: {count}")
            except Exception as e:
                logger.warning(f"   ⚠️  No se pudo contar registros: {e}")
            
            # Mostrar estructura de columnas
            try:
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position;
                """)
                columns = cursor.fetchall()
                
                logger.info(f"   📋 Columnas ({len(columns)}):")
                for col in columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    logger.info(f"      - {col[0]}: {col[1]} ({nullable})")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  No se pudo obtener estructura: {e}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al verificar tablas: {e}")
        return False

def main():
    """Función principal."""
    logger.info("🔍 Verificando tablas existentes en Supabase...")
    logger.info("=" * 60)
    
    success = check_existing_tables()
    
    if success:
        logger.info("\n🎉 Verificación completada")
    else:
        logger.error("❌ Error en la verificación")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 