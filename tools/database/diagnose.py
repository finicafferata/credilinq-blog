#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from supabase import create_client

def diagnose_database():
    """Diagnostica el estado actual de la base de datos"""
    print("🔍 Diagnóstico Detallado de la Base de Datos Supabase\n")
    
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    print(f"📡 Conectando a: {url}")
    print(f"🔑 Usando clave: {key[:15]}..." if key else "❌ No hay clave")
    print()
    
    try:
        supabase = create_client(url, key)
        
        # Test 1: Verificar si podemos hacer una consulta básica
        print("🧪 Test 1: Consulta básica al esquema público")
        try:
            # Intentar listar tablas del esquema público
            response = supabase.rpc('exec', {
                'sql': """
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
                """
            }).execute()
            print("  ✅ Acceso al esquema público: OK")
            tables = response.data
            print(f"  📊 Tablas encontradas: {len(tables)}")
            for table in tables:
                print(f"    - {table['table_name']} ({table['table_type']})")
        except Exception as e:
            print(f"  ❌ Error accediendo al esquema público: {e}")
        
        print()
        
        # Test 2: Verificar tabla blog_posts específicamente
        print("🧪 Test 2: Acceso a tabla blog_posts")
        try:
            response = supabase.table("blog_posts").select("count").execute()
            print("  ✅ Acceso a blog_posts: OK")
            print(f"  📊 Respuesta: {response}")
        except Exception as e:
            print(f"  ❌ Error accediendo a blog_posts: {e}")
            error_details = str(e)
            if "permission denied" in error_details.lower():
                print("  💡 Problema de permisos detectado")
            elif "relation does not exist" in error_details.lower():
                print("  💡 La tabla no existe")
        
        print()
        
        # Test 3: Verificar RLS y políticas
        print("🧪 Test 3: Estado de Row Level Security")
        try:
            response = supabase.rpc('exec', {
                'sql': """
                SELECT 
                    schemaname, 
                    tablename, 
                    rowsecurity 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                    AND tablename IN ('blog_posts', 'documents', 'document_chunks', 'campaign', 'campaign_task');
                """
            }).execute()
            print("  ✅ Consulta RLS: OK")
            tables_rls = response.data
            for table in tables_rls:
                rls_status = "🔒 HABILITADO" if table['rowsecurity'] else "🔓 DESHABILITADO"
                print(f"    - {table['tablename']}: RLS {rls_status}")
        except Exception as e:
            print(f"  ❌ Error verificando RLS: {e}")
        
        print()
        
        # Test 4: Verificar políticas
        print("🧪 Test 4: Políticas de seguridad")
        try:
            response = supabase.rpc('exec', {
                'sql': """
                SELECT 
                    schemaname, 
                    tablename, 
                    policyname, 
                    cmd,
                    qual,
                    with_check
                FROM pg_policies 
                WHERE schemaname = 'public' 
                ORDER BY tablename, policyname;
                """
            }).execute()
            print("  ✅ Consulta políticas: OK")
            policies = response.data
            if policies:
                print(f"  📋 Políticas encontradas: {len(policies)}")
                current_table = None
                for policy in policies:
                    if policy['tablename'] != current_table:
                        current_table = policy['tablename']
                        print(f"    📊 Tabla: {current_table}")
                    print(f"      - {policy['policyname']} ({policy['cmd']})")
            else:
                print("  ⚠️  No se encontraron políticas")
        except Exception as e:
            print(f"  ❌ Error verificando políticas: {e}")
        
        print()
        
        # Test 5: Verificar extensiones
        print("🧪 Test 5: Extensiones instaladas")
        try:
            response = supabase.rpc('exec', {
                'sql': """
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname IN ('vector', 'uuid-ossp');
                """
            }).execute()
            print("  ✅ Consulta extensiones: OK")
            extensions = response.data
            if extensions:
                for ext in extensions:
                    print(f"    - {ext['extname']} v{ext['extversion']}")
            else:
                print("  ⚠️  No se encontraron las extensiones esperadas")
        except Exception as e:
            print(f"  ❌ Error verificando extensiones: {e}")
        
    except Exception as e:
        print(f"❌ Error fatal conectando a Supabase: {e}")
        return False
    
    print("\n" + "="*60)
    print("📋 RESUMEN DEL DIAGNÓSTICO:")
    print("Si ves errores de 'permission denied':")
    print("  1. Las tablas existen pero RLS está bloqueando el acceso")
    print("  2. Necesitas usar la clave de servicio O configurar políticas")
    print("\nSi ves 'relation does not exist':")
    print("  1. Las tablas no se crearon correctamente")
    print("  2. Necesitas ejecutar el script SQL de nuevo")
    print("\n🔧 Próximos pasos basados en los resultados...")
    
    return True

if __name__ == "__main__":
    diagnose_database() 