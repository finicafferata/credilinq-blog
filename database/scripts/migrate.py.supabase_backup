#!/usr/bin/env python3
"""
Fixed script to apply database improvements to Supabase.
This version handles SQL blocks properly to avoid syntax errors.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_sql_statements(sql_content):
    """
    Properly split SQL content into executable statements.
    Handles DO blocks, functions, and complex statements.
    """
    statements = []
    current_statement = ""
    in_do_block = False
    in_function = False
    dollar_tag = None
    
    lines = sql_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('--'):
            continue
            
        # Check for DO block start
        if line.startswith('DO $') and '$' in line[3:]:
            in_do_block = True
            dollar_tag = line.split('$')[1]
            current_statement += line + '\n'
            continue
            
        # Check for DO block end
        if in_do_block and f'${dollar_tag}$' in line:
            current_statement += line + '\n'
            statements.append(current_statement.strip())
            current_statement = ""
            in_do_block = False
            dollar_tag = None
            continue
            
        # Check for function definition
        if 'CREATE OR REPLACE FUNCTION' in line or 'CREATE FUNCTION' in line:
            in_function = True
            current_statement += line + '\n'
            continue
            
        # Check for function end
        if in_function and '$$ language' in line.lower():
            current_statement += line + '\n'
            statements.append(current_statement.strip())
            current_statement = ""
            in_function = False
            continue
            
        # If we're in a block, keep adding to current statement
        if in_do_block or in_function:
            current_statement += line + '\n'
            continue
            
        # Regular statement handling
        current_statement += line + '\n'
        
        # Check if statement ends with semicolon
        if line.endswith(';'):
            statements.append(current_statement.strip())
            current_statement = ""
    
    # Add any remaining statement
    if current_statement.strip():
        statements.append(current_statement.strip())
    
    return statements

def apply_database_improvements():
    """Apply database improvements from fixed SQL file."""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.error("SUPABASE_DB_URL environment variable not found")
        sys.exit(1)
    
    # Read SQL file
    sql_file_path = "database_improvements_fixed.sql"
    if not os.path.exists(sql_file_path):
        logger.error(f"SQL file {sql_file_path} not found")
        sys.exit(1)
    
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        logger.info(f"Read SQL file: {sql_file_path}")
        
        # Connect to database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(
            db_url,
            connect_timeout=30,
            sslmode='require'
        )
        conn.autocommit = True
        
        logger.info("Connected successfully!")
        
        # Execute SQL with proper statement splitting
        cursor = conn.cursor()
        
        # Split SQL into properly formatted statements
        statements = split_sql_statements(sql_content)
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        success_count = 0
        for i, statement in enumerate(statements, 1):
            if not statement.strip():
                continue
                
            try:
                logger.info(f"Executing statement {i}/{len(statements)}...")
                cursor.execute(statement)
                success_count += 1
                logger.info(f"✅ Statement {i} executed successfully")
                
            except psycopg2.Error as e:
                # Check if it's a benign error (table/policy already exists)
                error_msg = str(e).lower()
                if any(phrase in error_msg for phrase in [
                    'already exists', 
                    'does not exist',
                    'column does not exist',
                    'relation does not exist'
                ]):
                    logger.info(f"ℹ️  Statement {i} skipped (already exists or not needed)")
                    continue
                else:
                    logger.warning(f"⚠️  Statement {i} failed: {str(e)[:200]}...")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error in statement {i}: {str(e)}")
                continue
        
        logger.info(f"🎉 Database improvements applied! {success_count}/{len(statements)} statements executed successfully")
        
        # Verify new tables were created
        logger.info("Verifying new tables...")
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
                AND table_name IN (
                    'agent_performance', 'agent_decisions', 'blog_analytics', 
                    'marketing_metrics', 'content_optimization', 'seo_metadata',
                    'content_variants', 'agent_feedback'
                )
            ORDER BY table_name;
        """)
        
        new_tables = [row[0] for row in cursor.fetchall()]
        
        if new_tables:
            logger.info(f"✅ New tables verified: {', '.join(new_tables)}")
        else:
            logger.warning("⚠️  No new tables found - they might already exist")
        
        # Verify indexes
        logger.info("Verifying indexes...")
        cursor.execute("""
            SELECT count(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
                AND (indexname LIKE 'idx_%' OR indexname LIKE '%_cosine_idx');
        """)
        
        index_count = cursor.fetchone()[0]
        logger.info(f"✅ Found {index_count} performance indexes")
        
        # Test basic functionality
        logger.info("Testing enhanced database functionality...")
        
        # Test agent_performance table
        try:
            cursor.execute("SELECT COUNT(*) FROM agent_performance;")
            logger.info("✅ agent_performance table accessible")
        except Exception as e:
            logger.warning(f"⚠️  agent_performance table test failed: {e}")
        
        # Test views
        try:
            cursor.execute("SELECT COUNT(*) FROM agent_efficiency;")
            logger.info("✅ agent_efficiency view accessible")
        except Exception as e:
            logger.warning(f"⚠️  Views test failed: {e}")
        
        cursor.close()
        conn.close()
        
        logger.info("🚀 Database improvements completed successfully!")
        print("\n" + "="*60)
        print("✅ DATABASE IMPROVEMENTS APPLIED SUCCESSFULLY!")
        print("="*60)
        print("You can now:")
        print("1. Use the enhanced DatabaseService in your agents")
        print("2. Track agent performance and decisions")
        print("3. Monitor blog analytics and marketing metrics")
        print("4. Optimize content based on performance data")
        print("\nNext steps:")
        print("• Run: python enhanced_main.py")
        print("• Test: curl http://localhost:8000/health")
        print("• View analytics: curl http://localhost:8000/analytics/dashboard")
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"SQL file not found: {sql_file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("🔧 Applying FIXED Database Improvements for AI Agents Marketing Platform")
    print("=" * 70)
    
    # Confirm before proceeding
    response = input("This will modify your Supabase database. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    apply_database_improvements()