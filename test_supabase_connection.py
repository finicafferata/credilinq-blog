#!/usr/bin/env python3
"""
Script para diagnosticar problemas específicos de conexión a Supabase
"""
import os
import psycopg2
from dotenv import load_dotenv
from supabase import create_client

def test_supabase_api():
    """Prueba la conexión a la API de Supabase"""
    print("🧪 Probando conexión a Supabase API...")
    try:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ Variables SUPABASE_URL o SUPABASE_KEY no encontradas")
            return False
            
        client = create_client(url, key)
        
        # Probar una operación simple
        response = client.table("documents").select("*").limit(1).execute()
        print(f"✅ Conexión API exitosa. Respuesta: {len(response.data) if response.data else 0} registros")
        return True
        
    except Exception as e:
        print(f"❌ Error en conexión API: {str(e)}")
        return False

def test_database_connection():
    """Prueba la conexión directa a PostgreSQL"""
    print("\n🧪 Probando conexión directa a PostgreSQL...")
    try:
        load_dotenv()
        db_url = os.getenv("SUPABASE_DB_URL")
        
        if not db_url:
            print("❌ Variable SUPABASE_DB_URL no encontrada")
            return False
            
        print(f"📡 Intentando conectar a: {db_url[:50]}...")
        
        # Intentar conexión con timeout
        conn = psycopg2.connect(db_url, connect_timeout=10)
        cursor = conn.cursor()
        
        # Probar una consulta simple
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Conexión PostgreSQL exitosa. Versión: {version[0][:50]}...")
        
        # Verificar si existen las tablas necesarias
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('documents', 'document_chunks');
        """)
        tables = cursor.fetchall()
        print(f"📊 Tablas encontradas: {[t[0] for t in tables]}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Error de conexión PostgreSQL: {str(e)}")
        
        # Analizar el tipo de error
        error_msg = str(e).lower()
        if "network is unreachable" in error_msg:
            print("💡 Problema de red. Posibles causas:")
            print("   - Conexión a internet limitada")
            print("   - Firewall bloqueando la conexión")
            print("   - Proyecto Supabase suspendido/inactivo")
        elif "authentication failed" in error_msg:
            print("💡 Problema de autenticación:")
            print("   - Usuario o contraseña incorrectos en SUPABASE_DB_URL")
        elif "does not exist" in error_msg:
            print("💡 Problema de base de datos:")
            print("   - La base de datos no existe")
            print("   - URL incorrecta")
            
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        return False

def test_network_connectivity():
    """Prueba conectividad básica de red"""
    print("\n🌐 Probando conectividad de red...")
    try:
        import socket
        import urllib.request
        
        # Extraer el host de la URL
        load_dotenv()
        db_url = os.getenv("SUPABASE_DB_URL", "")
        
        if "db." in db_url:
            # Extraer el host de la URL de PostgreSQL
            host = db_url.split("@")[1].split(":")[0] if "@" in db_url else ""
            if host:
                print(f"🔍 Probando conectividad a: {host}")
                
                # Ping básico
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, 5432))
                sock.close()
                
                if result == 0:
                    print("✅ Puerto 5432 es accesible")
                else:
                    print("❌ Puerto 5432 no es accesible")
                    
        # Probar conexión HTTP general
        try:
            urllib.request.urlopen("https://supabase.com", timeout=5)
            print("✅ Conectividad HTTP a Supabase.com exitosa")
        except:
            print("❌ Sin conectividad HTTP a Supabase.com")
            
    except Exception as e:
        print(f"❌ Error en prueba de red: {str(e)}")

def main():
    print("🔧 DIAGNÓSTICO COMPLETO DE SUPABASE")
    print("=" * 60)
    
    # Ejecutar todas las pruebas
    api_ok = test_supabase_api()
    db_ok = test_database_connection()
    test_network_connectivity()
    
    print("\n" + "=" * 60)
    print("📋 RESUMEN:")
    print(f"   API Supabase: {'✅' if api_ok else '❌'}")
    print(f"   PostgreSQL: {'✅' if db_ok else '❌'}")
    
    if not api_ok and not db_ok:
        print("\n🚨 PROBLEMA CRÍTICO:")
        print("   No se puede conectar a Supabase por ningún método")
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Verifica que tu proyecto Supabase esté activo")
        print("   2. Revisa tu conexión a internet")
        print("   3. Considera usar la alternativa temporal sin Supabase")
    elif api_ok and not db_ok:
        print("\n⚠️  PROBLEMA PARCIAL:")
        print("   API funciona pero PostgreSQL no. Verifica SUPABASE_DB_URL")
    elif db_ok:
        print("\n✅ CONEXIÓN EXITOSA:")
        print("   El problema puede estar en otro lugar del código")

if __name__ == "__main__":
    main() 