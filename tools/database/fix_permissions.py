#!/usr/bin/env python3
"""
Quick permission fix for Supabase database.
Attempts to grant necessary permissions programmatically.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_permissions():
    """Try to fix database permissions programmatically."""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.error("SUPABASE_DB_URL environment variable not found")
        sys.exit(1)
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(
            db_url,
            connect_timeout=30,
            sslmode='require'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("Connected successfully!")
        
        # Check current user
        cursor.execute("SELECT current_user, session_user;")
        current_user, session_user = cursor.fetchone()
        logger.info(f"Current user: {current_user}, Session user: {session_user}")
        
        # Check if we have superuser privileges
        cursor.execute("SELECT usesuper FROM pg_user WHERE usename = current_user;")
        is_superuser = cursor.fetchone()[0] if cursor.rowcount > 0 else False
        logger.info(f"Superuser privileges: {is_superuser}")
        
        # Try to grant basic permissions
        permissions_to_grant = [
            "GRANT USAGE ON SCHEMA public TO authenticated;",
            "GRANT USAGE ON SCHEMA public TO anon;", 
            "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO authenticated;",
            "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO authenticated;",
        ]
        
        success_count = 0
        for permission in permissions_to_grant:
            try:
                cursor.execute(permission)
                success_count += 1
                logger.info(f"✅ Granted: {permission.split()[1:4]}")
            except psycopg2.Error as e:
                if "permission denied" in str(e).lower():
                    logger.warning(f"⚠️  Permission denied for: {permission.split()[1:4]}")
                else:
                    logger.warning(f"⚠️  Failed: {permission.split()[1:4]} - {str(e)[:100]}")
        
        if success_count > 0:
            logger.info(f"✅ Successfully granted {success_count}/{len(permissions_to_grant)} permissions")
        else:
            logger.warning("❌ Could not grant any permissions programmatically")
        
        # Test basic table access
        logger.info("Testing table access...")
        
        test_tables = ["blog_posts", "documents", "campaign"]
        accessible_tables = []
        
        for table in test_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                accessible_tables.append(table)
                logger.info(f"✅ {table}: {count} rows")
            except psycopg2.Error as e:
                logger.warning(f"❌ {table}: {str(e)[:100]}")
        
        cursor.close()
        conn.close()
        
        # Provide recommendations
        print("\n" + "="*60)
        print("PERMISSION FIX RESULTS")
        print("="*60)
        
        if len(accessible_tables) >= 2:
            print("✅ GOOD NEWS: Basic table access is working!")
            print("✅ Your application should function with existing tables.")
        else:
            print("❌ PERMISSION ISSUES DETECTED")
            print("❌ Limited table access detected.")
        
        print(f"\nAccessible tables: {', '.join(accessible_tables) if accessible_tables else 'None'}")
        
        if success_count == 0:
            print("\n🔧 MANUAL FIX REQUIRED:")
            print("1. Go to your Supabase Dashboard")
            print("2. Open SQL Editor")
            print("3. Run the contents of 'fix_database_permissions.sql'")
            print("4. Make sure you're using a role with sufficient privileges")
        
        print("\n📋 NEXT STEPS:")
        print("• Test health endpoint: curl http://localhost:8000/health")
        print("• If still failing, check Supabase RLS policies")
        print("• Consider using service_role key instead of anon key")
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        print("\n❌ DATABASE CONNECTION FAILED")
        print("This might be due to:")
        print("• Incorrect SUPABASE_DB_URL")
        print("• Network connectivity issues") 
        print("• Database credentials")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("🔧 Quick Database Permission Fix")
    print("=" * 40)
    fix_permissions()