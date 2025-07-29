# 🔧 Solución a Errores de Conexión Backend

## 🚨 Problema Actual
El backend falla con el error: `psycopg2.OperationalError: connection to server at "db.ptrbrrbxojmpwcnryryy.supabase.co" failed: Network is unreachable`

**Causa:** Faltan las variables de entorno para conectarse a Supabase.

## 📋 Solución Completa

### Paso 1: Crear archivo .env

Crea un archivo llamado `.env` en la raíz del proyecto con este contenido:

```bash
# Environment Variables - Configura con tus claves reales

# OpenAI API Key (requerida para embeddings y LLM)
OPENAI_API_KEY="your-openai-api-key-here"

# Google AI API Key (opcional, para Gemini LLM)
GOOGLE_API_KEY="your-google-api-key-here"

# Supabase Configuration (requerida para funcionalidad RAG)
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-supabase-anon-key-here"
SUPABASE_DB_URL="postgresql://postgres:tu-password@db.tu-project-id.supabase.co:5432/postgres"
SUPABASE_STORAGE_BUCKET="documents"
```

### Paso 2: Obtener Credenciales de Supabase

1. **Crear cuenta en Supabase:**
   - Ve a [supabase.com](https://supabase.com)
   - Crea una cuenta gratis
   - Crea un nuevo proyecto

2. **Obtener credenciales:**
   - En tu dashboard, ve a **Settings > API**
   - Copia **Project URL** → `SUPABASE_URL`
   - Copia **anon public key** → `SUPABASE_KEY`
   - Ve a **Settings > Database**
   - Copia **Connection string** → `SUPABASE_DB_URL`

3. **Configurar la base de datos:**
   - En tu proyecto Supabase, ve a **SQL Editor**
   - Ejecuta este SQL para crear las tablas necesarias:

```sql
-- Crear tabla de documentos
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Crear tabla de chunks con embeddings (requiere extensión pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536) -- OpenAI embeddings son de 1536 dimensiones
);

-- Crear índice para búsquedas por similaridad
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops);

-- Crear bucket de storage
INSERT INTO storage.buckets (id, name, public) VALUES ('documents', 'documents', false);
```

### Paso 3: Verificar Configuración

Ejecuta el script de verificación:

```bash
python check_env.py
```

### Paso 4: Reiniciar el Backend

Una vez configurado todo:

```bash
# Si usas uvicorn directamente
uvicorn main:api --reload

# O si tienes otro comando de inicio
python main.py
```

## 🚀 Alternativa Temporal (Sin Supabase)

Si quieres probar la aplicación sin configurar Supabase inmediatamente, puedo modificar el código para usar archivos locales temporalmente. Esta opción:

- ✅ Permite que la aplicación funcione inmediatamente
- ✅ Usa los documentos en la carpeta `knowledge_base/`
- ❌ No persiste documentos subidos
- ❌ Funcionalidad limitada de RAG

**¿Quieres que implemente esta alternativa temporal?**

## 🔍 Diagnóstico de Problemas

### Verificar variables de entorno:
```bash
python check_env.py
```

### Verificar conectividad a Supabase:
```python
# Prueba rápida en Python
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
client = create_client(url, key)
print("✅ Conexión a Supabase exitosa")
```

### Errores Comunes:

1. **Network is unreachable**: Variables de entorno no configuradas
2. **Authentication failed**: Claves incorrectas
3. **Database does not exist**: Tablas no creadas
4. **Permission denied**: Políticas RLS muy restrictivas

## 📞 Soporte

Si sigues teniendo problemas:
1. Ejecuta `python check_env.py` y comparte el resultado
2. Verifica que tu proyecto Supabase esté activo
3. Asegúrate de que las tablas estén creadas correctamente 