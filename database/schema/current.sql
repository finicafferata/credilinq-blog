-- =====================================================
-- SCRIPT COMPLETO: Configuración de Base de Datos desde Cero
-- Ejecuta este script completo en el SQL Editor de Supabase
-- =====================================================

-- ===== PASO 1: LIMPIAR TABLAS EXISTENTES (si existen) =====
-- Eliminar políticas existentes primero
DROP POLICY IF EXISTS "Permitir lectura pública de blog_posts" ON blog_posts;
DROP POLICY IF EXISTS "Permitir inserción pública de blog_posts" ON blog_posts;
DROP POLICY IF EXISTS "Permitir actualización pública de blog_posts" ON blog_posts;
DROP POLICY IF EXISTS "Permitir eliminación pública de blog_posts" ON blog_posts;

DROP POLICY IF EXISTS "Permitir lectura pública de documents" ON documents;
DROP POLICY IF EXISTS "Permitir inserción pública de documents" ON documents;
DROP POLICY IF EXISTS "Permitir actualización pública de documents" ON documents;
DROP POLICY IF EXISTS "Permitir eliminación pública de documents" ON documents;

DROP POLICY IF EXISTS "Permitir lectura pública de document_chunks" ON document_chunks;
DROP POLICY IF EXISTS "Permitir inserción pública de document_chunks" ON document_chunks;
DROP POLICY IF EXISTS "Permitir actualización pública de document_chunks" ON document_chunks;
DROP POLICY IF EXISTS "Permitir eliminación pública de document_chunks" ON document_chunks;

DROP POLICY IF EXISTS "Permitir lectura pública de campaign" ON campaign;
DROP POLICY IF EXISTS "Permitir inserción pública de campaign" ON campaign;
DROP POLICY IF EXISTS "Permitir actualización pública de campaign" ON campaign;
DROP POLICY IF EXISTS "Permitir eliminación pública de campaign" ON campaign;

DROP POLICY IF EXISTS "Permitir lectura pública de campaign_task" ON campaign_task;
DROP POLICY IF EXISTS "Permitir inserción pública de campaign_task" ON campaign_task;
DROP POLICY IF EXISTS "Permitir actualización pública de campaign_task" ON campaign_task;
DROP POLICY IF EXISTS "Permitir eliminación pública de campaign_task" ON campaign_task;

-- Eliminar tablas (en orden correcto para evitar errores de foreign key)
DROP TABLE IF EXISTS campaign_task;
DROP TABLE IF EXISTS campaign;
DROP TABLE IF EXISTS document_chunks;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS blog_posts;

-- ===== PASO 2: HABILITAR EXTENSIONES =====
-- Extensión vector para embeddings (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- ===== PASO 3: CREAR TABLAS =====

-- Tabla blog_posts
CREATE TABLE "blog_posts" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "title" TEXT NOT NULL,
    "content_markdown" TEXT NOT NULL,
    "initial_prompt" JSONB,
    "status" TEXT DEFAULT 'draft',
    "created_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "blog_posts_pkey" PRIMARY KEY ("id")
);

-- Tabla documents
CREATE TABLE "documents" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "title" TEXT NOT NULL,
    "storage_path" TEXT NOT NULL,
    "uploaded_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
);

-- Tabla document_chunks
CREATE TABLE "document_chunks" (
    "id" BIGSERIAL NOT NULL,
    "document_id" UUID,
    "content" TEXT NOT NULL,
    "embedding" vector(1536), -- OpenAI embeddings son de 1536 dimensiones
    CONSTRAINT "document_chunks_pkey" PRIMARY KEY ("id")
);

-- Tabla campaign
CREATE TABLE "campaign" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "blog_id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "campaign_pkey" PRIMARY KEY ("id")
);

-- Tabla campaign_task
CREATE TABLE "campaign_task" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "campaign_id" UUID NOT NULL,
    "taskType" TEXT NOT NULL,
    "targetFormat" TEXT,
    "targetAsset" TEXT,
    "status" TEXT NOT NULL,
    "result" TEXT,
    "imageUrl" TEXT,
    "error" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "campaign_task_pkey" PRIMARY KEY ("id")
);

-- ===== PASO 4: CREAR ÍNDICES =====
-- Índice para búsquedas por similaridad en embeddings
CREATE INDEX "document_chunks_embedding_idx" ON "document_chunks" USING ivfflat ("embedding" vector_cosine_ops);

-- Índice único para campaign
CREATE UNIQUE INDEX "campaign_blog_id_key" ON "campaign"("blog_id");

-- ===== PASO 5: AGREGAR FOREIGN KEYS =====
ALTER TABLE "document_chunks" ADD CONSTRAINT "document_chunks_document_id_fkey" 
    FOREIGN KEY ("document_id") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE NO ACTION;

ALTER TABLE "campaign" ADD CONSTRAINT "campaign_blog_id_fkey" 
    FOREIGN KEY ("blog_id") REFERENCES "blog_posts"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "campaign_task" ADD CONSTRAINT "campaign_task_campaign_id_fkey" 
    FOREIGN KEY ("campaign_id") REFERENCES "campaign"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ===== PASO 6: CONFIGURAR ROW LEVEL SECURITY (RLS) =====
-- Habilitar RLS en todas las tablas
ALTER TABLE blog_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE campaign ENABLE ROW LEVEL SECURITY;
ALTER TABLE campaign_task ENABLE ROW LEVEL SECURITY;

-- ===== PASO 7: CREAR POLÍTICAS DE ACCESO PÚBLICO =====
-- NOTA: Estas políticas permiten acceso público total para desarrollo
-- En producción, considera políticas más restrictivas

-- Políticas para blog_posts
CREATE POLICY "Permitir lectura pública de blog_posts" ON blog_posts
    FOR SELECT USING (true);

CREATE POLICY "Permitir inserción pública de blog_posts" ON blog_posts
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Permitir actualización pública de blog_posts" ON blog_posts
    FOR UPDATE USING (true);

CREATE POLICY "Permitir eliminación pública de blog_posts" ON blog_posts
    FOR DELETE USING (true);

-- Políticas para documents
CREATE POLICY "Permitir lectura pública de documents" ON documents
    FOR SELECT USING (true);

CREATE POLICY "Permitir inserción pública de documents" ON documents
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Permitir actualización pública de documents" ON documents
    FOR UPDATE USING (true);

CREATE POLICY "Permitir eliminación pública de documents" ON documents
    FOR DELETE USING (true);

-- Políticas para document_chunks
CREATE POLICY "Permitir lectura pública de document_chunks" ON document_chunks
    FOR SELECT USING (true);

CREATE POLICY "Permitir inserción pública de document_chunks" ON document_chunks
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Permitir actualización pública de document_chunks" ON document_chunks
    FOR UPDATE USING (true);

CREATE POLICY "Permitir eliminación pública de document_chunks" ON document_chunks
    FOR DELETE USING (true);

-- Políticas para campaign
CREATE POLICY "Permitir lectura pública de campaign" ON campaign
    FOR SELECT USING (true);

CREATE POLICY "Permitir inserción pública de campaign" ON campaign
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Permitir actualización pública de campaign" ON campaign
    FOR UPDATE USING (true);

CREATE POLICY "Permitir eliminación pública de campaign" ON campaign
    FOR DELETE USING (true);

-- Políticas para campaign_task
CREATE POLICY "Permitir lectura pública de campaign_task" ON campaign_task
    FOR SELECT USING (true);

CREATE POLICY "Permitir inserción pública de campaign_task" ON campaign_task
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Permitir actualización pública de campaign_task" ON campaign_task
    FOR UPDATE USING (true);

CREATE POLICY "Permitir eliminación pública de campaign_task" ON campaign_task
    FOR DELETE USING (true);

-- ===== PASO 8: CONFIGURAR STORAGE BUCKET =====
-- Crear bucket para documentos (si no existe)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('documents', 'documents', false)
ON CONFLICT (id) DO NOTHING;

-- Política de storage para documentos
CREATE POLICY "Permitir subida pública de documentos" ON storage.objects
FOR INSERT WITH CHECK (bucket_id = 'documents');

CREATE POLICY "Permitir lectura pública de documentos" ON storage.objects
FOR SELECT USING (bucket_id = 'documents');

CREATE POLICY "Permitir actualización pública de documentos" ON storage.objects
FOR UPDATE USING (bucket_id = 'documents');

CREATE POLICY "Permitir eliminación pública de documentos" ON storage.objects
FOR DELETE USING (bucket_id = 'documents');

-- ===== PASO 9: VERIFICACIÓN =====
-- Verificar que las tablas se crearon correctamente
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name IN ('blog_posts', 'documents', 'document_chunks', 'campaign', 'campaign_task')
ORDER BY table_name;

-- Verificar que las políticas se crearon correctamente
SELECT 
    schemaname, 
    tablename, 
    policyname, 
    permissive, 
    roles, 
    cmd
FROM pg_policies 
WHERE schemaname = 'public' 
ORDER BY tablename, policyname;

-- ===== FINALIZADO =====
-- ✅ Base de datos configurada completamente
-- ✅ Todas las tablas creadas
-- ✅ RLS habilitado con políticas de acceso público
-- ✅ Storage bucket configurado
-- ✅ Extensión vector habilitada
-- 
-- Ahora puedes usar tu clave anónima de Supabase sin problemas! 