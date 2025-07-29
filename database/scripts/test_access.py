#!/usr/bin/env python3
"""
Test database access after permission fixes.
This will help verify which specific access method works.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import psycopg2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_access():
    """Test different ways to access the database."""
    
    load_dotenv()
    
    print("🧪 Testing Database Access Methods")
    print("=" * 50)
    
    # Test 1: Direct database connection
    print("\n1️⃣  Testing Direct Database Connection...")
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            print("❌ SUPABASE_DB_URL not found")
        else:
            conn = psycopg2.connect(db_url, connect_timeout=10, sslmode='require')
            conn.autocommit = True
            cur = conn.cursor()
            
            # Test basic query
            cur.execute("SELECT current_user, version();")
            user, version = cur.fetchone()
            print(f"✅ Connected as: {user}")
            print(f"✅ PostgreSQL version: {version[:50]}...")
            
            # Test table access
            test_tables = ['blog_posts', 'agent_performance', 'blog_analytics']
            for table in test_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()[0]
                    print(f"✅ {table}: {count} rows")
                except Exception as e:
                    print(f"❌ {table}: {str(e)[:80]}...")
            
            cur.close()
            conn.close()
            
    except Exception as e:
        print(f"❌ Direct connection failed: {str(e)}")
    
    # Test 2: Supabase client with anon key
    print("\n2️⃣  Testing Supabase Client (Anon Key)...")
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ SUPABASE_URL or SUPABASE_KEY not found")
        else:
            supabase = create_client(supabase_url, supabase_key)
            
            # Test basic table access
            try:
                response = supabase.table("blog_posts").select("id").limit(1).execute()
                print("✅ blog_posts table accessible")
                print(f"✅ Response count: {len(response.data)}")
            except Exception as e:
                print(f"❌ blog_posts access failed: {str(e)}")
            
            # Test analytics tables
            analytics_tables = ['agent_performance', 'blog_analytics', 'marketing_metrics']
            for table in analytics_tables:
                try:
                    response = supabase.table(table).select("id").limit(1).execute()
                    print(f"✅ {table}: accessible")
                except Exception as e:
                    error_msg = str(e)
                    if "does not exist" in error_msg:
                        print(f"⚠️  {table}: table doesn't exist")
                    elif "permission denied" in error_msg:
                        print(f"❌ {table}: permission denied")
                    else:
                        print(f"❌ {table}: {error_msg[:60]}...")
    
    except Exception as e:
        print(f"❌ Supabase client failed: {str(e)}")
    
    # Test 3: Check environment variables
    print("\n3️⃣  Environment Variables Check...")
    env_vars = [
        "SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_DB_URL", 
        "OPENAI_API_KEY", "DATABASE_URL"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Show first/last few chars for security
            masked = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***"
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test 4: Check key type
    print("\n4️⃣  Analyzing Supabase Key Type...")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    if supabase_key:
        if supabase_key.startswith("eyJ"):
            # Try to decode JWT to see role
            try:
                import base64
                import json
                
                # JWT has 3 parts separated by dots
                parts = supabase_key.split('.')
                if len(parts) >= 2:
                    # Decode payload (second part)
                    payload = parts[1]
                    # Add padding if needed
                    padding = 4 - len(payload) % 4
                    if padding != 4:
                        payload += '=' * padding
                    
                    decoded = base64.b64decode(payload)
                    jwt_data = json.loads(decoded)
                    
                    role = jwt_data.get('role', 'unknown')
                    iss = jwt_data.get('iss', 'unknown')
                    
                    print(f"✅ Key type: JWT")
                    print(f"✅ Role: {role}")
                    print(f"✅ Issuer: {iss}")
                    
                    if role == 'anon':
                        print("ℹ️  Using anonymous key - limited permissions")
                        print("💡 Consider using service_role key for full access")
                    elif role == 'service_role':
                        print("✅ Using service role key - should have full access")
                    
            except Exception as e:
                print(f"⚠️  Could not decode JWT: {str(e)}")
        else:
            print("⚠️  Key doesn't look like a JWT")
    
    print("\n📋 RECOMMENDATIONS:")
    print("=" * 50)
    print("1. If direct DB works but Supabase client doesn't:")
    print("   → Run fix_supabase_rls.sql in Supabase SQL Editor")
    print("2. If using 'anon' role:")
    print("   → Switch to 'service_role' key for development")
    print("3. If tables don't exist:")
    print("   → Run python apply_database_improvements_fixed.py")
    print("4. Check Supabase Dashboard → Settings → API for correct keys")

if __name__ == "__main__":
    test_database_access()