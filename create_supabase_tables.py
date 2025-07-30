#!/usr/bin/env python3
"""
Script para crear las tablas necesarias en Supabase.
"""

import psycopg2
import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Crear las tablas necesarias en Supabase."""
    load_dotenv()
    
    try:
        # Conectar a la base de datos
        database_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🔗 Conectado a Supabase")
        
        # Crear tabla blogs
        logger.info("📝 Creando tabla 'blogs'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blogs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title VARCHAR(255) NOT NULL,
                content_markdown TEXT,
                initial_prompt TEXT,
                status VARCHAR(50) DEFAULT 'draft',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                published_at TIMESTAMP WITH TIME ZONE,
                author VARCHAR(100),
                tags TEXT[],
                seo_title VARCHAR(255),
                seo_description TEXT,
                featured_image_url TEXT,
                read_time_minutes INTEGER DEFAULT 5,
                word_count INTEGER DEFAULT 0
            );
        """)
        
        # Crear tabla campaigns
        logger.info("📝 Creando tabla 'campaigns'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                blog_id UUID REFERENCES blogs(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'draft',
                platform VARCHAR(50),
                target_audience TEXT,
                budget DECIMAL(10,2),
                start_date DATE,
                end_date DATE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metrics JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        # Crear tabla campaign_tasks
        logger.info("📝 Creando tabla 'campaign_tasks'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaign_tasks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
                task_type VARCHAR(100) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'pending',
                priority VARCHAR(20) DEFAULT 'medium',
                assigned_to VARCHAR(100),
                due_date TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                result_data JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        # Crear tabla documents
        logger.info("📝 Creando tabla 'documents'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR(255) NOT NULL,
                content TEXT,
                file_type VARCHAR(50),
                file_size INTEGER,
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                processed_at TIMESTAMP WITH TIME ZONE,
                status VARCHAR(50) DEFAULT 'pending',
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        # Crear tabla analytics
        logger.info("📝 Creando tabla 'analytics'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                blog_id UUID REFERENCES blogs(id) ON DELETE CASCADE,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB DEFAULT '{}'::jsonb,
                user_agent TEXT,
                ip_address INET,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Crear índices para mejor rendimiento
        logger.info("📝 Creando índices...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blogs_status ON blogs(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blogs_created_at ON blogs(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaigns_blog_id ON campaigns(blog_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaign_tasks_campaign_id ON campaign_tasks(campaign_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_blog_id ON analytics(blog_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);")
        
        # Commit los cambios
        conn.commit()
        
        logger.info("✅ Todas las tablas creadas exitosamente")
        
        # Verificar las tablas creadas
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        logger.info(f"📋 Tablas disponibles ({len(tables)}):")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al crear las tablas: {e}")
        return False

def main():
    """Función principal."""
    logger.info("🚀 Creando tablas en Supabase...")
    logger.info("=" * 50)
    
    success = create_tables()
    
    if success:
        logger.info("🎉 ¡Base de datos configurada correctamente!")
        logger.info("💡 Ahora puedes ejecutar tu aplicación")
    else:
        logger.error("❌ Error al configurar la base de datos")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 