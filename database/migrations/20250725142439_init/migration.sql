CREATE EXTENSION IF NOT EXISTS vector;
-- CreateTable
CREATE TABLE "blog_posts" (
    "id" UUID NOT NULL,
    "title" TEXT NOT NULL,
    "content_markdown" TEXT NOT NULL,
    "initial_prompt" JSONB,
    "status" TEXT DEFAULT 'draft',
    "created_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "blog_posts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "document_chunks" (
    "id" BIGSERIAL NOT NULL,
    "document_id" UUID,
    "content" TEXT NOT NULL,
    "embedding" vector,

    CONSTRAINT "document_chunks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "documents" (
    "id" UUID NOT NULL,
    "title" TEXT NOT NULL,
    "storage_path" TEXT NOT NULL,
    "uploaded_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "campaign" (
    "id" UUID NOT NULL,
    "blog_id" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "campaign_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "campaign_task" (
    "id" UUID NOT NULL,
    "campaign_id" UUID NOT NULL,
    "taskType" TEXT NOT NULL,
    "targetFormat" TEXT,
    "targetAsset" TEXT,
    "status" TEXT NOT NULL,
    "result" TEXT,
    "imageUrl" TEXT,
    "error" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "campaign_task_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "document_chunks_embedding_idx" ON "document_chunks"("embedding");

-- CreateIndex
CREATE UNIQUE INDEX "campaign_blog_id_key" ON "campaign"("blog_id");

-- AddForeignKey
ALTER TABLE "document_chunks" ADD CONSTRAINT "document_chunks_document_id_fkey" FOREIGN KEY ("document_id") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE NO ACTION;

-- AddForeignKey
ALTER TABLE "campaign" ADD CONSTRAINT "campaign_blog_id_fkey" FOREIGN KEY ("blog_id") REFERENCES "blog_posts"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "campaign_task" ADD CONSTRAINT "campaign_task_campaign_id_fkey" FOREIGN KEY ("campaign_id") REFERENCES "campaign"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
