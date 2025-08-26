-- Migration: Convert existing campaign tasks to content deliverables
-- This migrates existing task data to the new content-first system
-- Date: 2024-12-19

-- Step 1: Create a function to convert task types to content types
CREATE OR REPLACE FUNCTION map_task_type_to_content_type(task_type TEXT)
RETURNS "ContentType" AS $$
BEGIN
    RETURN CASE 
        WHEN task_type IN ('content_creation', 'content_editing') THEN 'blog_post'::"ContentType"
        WHEN task_type = 'social_media_adaptation' THEN 'social_media_post'::"ContentType"
        WHEN task_type = 'email_formatting' THEN 'email_campaign'::"ContentType"
        WHEN task_type IN ('blog_to_linkedin', 'blog_to_twitter') THEN 'social_media_post'::"ContentType"
        WHEN task_type = 'blog_to_video_script' THEN 'video_script'::"ContentType"
        WHEN task_type = 'image_generation' THEN 'infographic_concept'::"ContentType"
        WHEN task_type = 'content_repurposing' THEN 'blog_post'::"ContentType"
        ELSE 'blog_post'::"ContentType" -- Default fallback
    END;
END;
$$ LANGUAGE plpgsql;

-- Step 2: Create a function to determine deliverable status from task status
CREATE OR REPLACE FUNCTION map_task_status_to_deliverable_status(task_status TEXT)
RETURNS "DeliverableStatus" AS $$
BEGIN
    RETURN CASE
        WHEN task_status IN ('completed', 'approved') THEN 'approved'::"DeliverableStatus"
        WHEN task_status = 'in_progress' THEN 'in_review'::"DeliverableStatus"
        WHEN task_status IN ('needs_review', 'rejected') THEN 'needs_revision'::"DeliverableStatus"
        WHEN task_status = 'pending' THEN 'draft'::"DeliverableStatus"
        ELSE 'draft'::"DeliverableStatus" -- Default fallback
    END;
END;
$$ LANGUAGE plpgsql;

-- Step 3: Create a function to extract title from task result
CREATE OR REPLACE FUNCTION extract_title_from_result(result TEXT, task_type TEXT)
RETURNS TEXT AS $$
BEGIN
    IF result IS NULL OR result = '' THEN
        RETURN CASE 
            WHEN task_type = 'content_creation' THEN 'Blog Post'
            WHEN task_type = 'social_media_adaptation' THEN 'Social Media Post'
            WHEN task_type = 'email_formatting' THEN 'Email Campaign'
            ELSE 'Content Piece'
        END;
    END IF;
    
    -- Try to extract first line as title if it looks like a title
    IF result ~ '^#\s+' THEN
        -- Extract markdown header
        RETURN TRIM(REGEXP_REPLACE(SPLIT_PART(result, E'\n', 1), '^#+\s*', ''));
    ELSIF LENGTH(result) > 10 AND POSITION(E'\n' IN result) > 0 THEN
        -- Use first line if reasonable length
        DECLARE
            first_line TEXT := SPLIT_PART(result, E'\n', 1);
        BEGIN
            IF LENGTH(first_line) BETWEEN 10 AND 200 THEN
                RETURN first_line;
            END IF;
        END;
    END IF;
    
    -- Fallback: generate descriptive title
    RETURN CASE 
        WHEN task_type = 'content_creation' THEN 'Generated Blog Post'
        WHEN task_type = 'social_media_adaptation' THEN 'Social Media Content'
        WHEN task_type = 'email_formatting' THEN 'Email Campaign Content'
        ELSE 'Content Deliverable'
    END;
END;
$$ LANGUAGE plpgsql;

-- Step 4: Create a function to extract key messages from task result
CREATE OR REPLACE FUNCTION extract_key_messages_from_result(result TEXT, target_format TEXT)
RETURNS TEXT[] AS $$
BEGIN
    -- For now, create default key messages based on content analysis
    -- In a more sophisticated version, we could use NLP to extract actual themes
    
    IF result IS NULL OR result = '' THEN
        RETURN ARRAY['Generated content'];
    END IF;
    
    -- Simple heuristic: look for bullet points or numbered lists
    DECLARE
        messages TEXT[] := '{}';
    BEGIN
        -- Extract lines that start with bullet points or numbers
        SELECT array_agg(DISTINCT TRIM(REGEXP_REPLACE(line, '^[•\-\*]\s*|^\d+\.\s*', '')))
        INTO messages
        FROM (
            SELECT UNNEST(STRING_TO_ARRAY(result, E'\n')) AS line
        ) lines
        WHERE line ~ '^[•\-\*]\s+\w+|^\d+\.\s+\w+'
        AND LENGTH(TRIM(REGEXP_REPLACE(line, '^[•\-\*]\s*|^\d+\.\s*', ''))) BETWEEN 10 AND 100
        LIMIT 5;
        
        -- If no bullet points found, create generic messages
        IF array_length(messages, 1) IS NULL OR array_length(messages, 1) = 0 THEN
            messages := CASE
                WHEN target_format = 'LinkedIn' THEN ARRAY['LinkedIn engagement', 'Professional insights']
                WHEN target_format = 'Twitter' THEN ARRAY['Social media content', 'Brand awareness']
                WHEN target_format = 'Email' THEN ARRAY['Email marketing', 'Customer engagement']
                ELSE ARRAY['Content marketing', 'Brand messaging']
            END;
        END IF;
        
        RETURN messages;
    END;
END;
$$ LANGUAGE plpgsql;

-- Step 5: Migrate existing campaign tasks to content deliverables
INSERT INTO content_deliverables (
    title,
    content,
    summary,
    content_type,
    format,
    status,
    campaign_id,
    narrative_order,
    key_messages,
    target_audience,
    tone,
    platform,
    word_count,
    reading_time,
    created_by,
    last_edited_by,
    version,
    is_published,
    created_at,
    updated_at,
    metadata
)
SELECT 
    extract_title_from_result(ct.result, ct.task_type::text) as title,
    COALESCE(ct.result, 'Content generated from task: ' || ct.task_type::text) as content,
    CASE 
        WHEN LENGTH(ct.result) > 200 THEN LEFT(ct.result, 200) || '...'
        ELSE ct.result
    END as summary,
    map_task_type_to_content_type(ct.task_type::text) as content_type,
    'markdown'::"ContentFormat" as format,
    map_task_status_to_deliverable_status(ct.status::text) as status,
    ct.campaign_id,
    ROW_NUMBER() OVER (PARTITION BY ct.campaign_id ORDER BY ct.created_at) as narrative_order,
    extract_key_messages_from_result(ct.result, ct.target_format) as key_messages,
    'Business professionals and decision makers' as target_audience,
    'professional' as tone,
    LOWER(COALESCE(ct.target_format, 'blog')) as platform,
    CASE 
        WHEN ct.result IS NOT NULL THEN array_length(string_to_array(ct.result, ' '), 1)
        ELSE NULL
    END as word_count,
    CASE 
        WHEN ct.result IS NOT NULL THEN GREATEST(1, array_length(string_to_array(ct.result, ' '), 1) / 250)
        ELSE NULL
    END as reading_time,
    'TaskMigrationAgent' as created_by,
    'TaskMigrationAgent' as last_edited_by,
    1 as version,
    CASE WHEN ct.status::text IN ('completed', 'approved') THEN true ELSE false END as is_published,
    ct.created_at,
    ct.updated_at,
    jsonb_build_object(
        'migrated_from_task', true,
        'original_task_id', ct.id,
        'original_task_type', ct.task_type,
        'original_status', ct.status,
        'migration_date', NOW()
    ) as metadata
FROM campaign_tasks ct
WHERE ct.result IS NOT NULL 
AND ct.result != ''
AND NOT EXISTS (
    -- Don't migrate tasks that already have linked deliverables
    SELECT 1 FROM content_deliverables cd WHERE cd.metadata->>'original_task_id' = ct.id::text
);

-- Step 6: Link the migrated deliverables back to their original tasks
UPDATE campaign_tasks 
SET deliverable_id = (
    SELECT cd.id 
    FROM content_deliverables cd 
    WHERE cd.metadata->>'original_task_id' = campaign_tasks.id::text
)
WHERE deliverable_id IS NULL
AND EXISTS (
    SELECT 1 
    FROM content_deliverables cd 
    WHERE cd.metadata->>'original_task_id' = campaign_tasks.id::text
);

-- Step 7: Create content narratives for campaigns that have deliverables
INSERT INTO content_narratives (
    campaign_id,
    title,
    description,
    narrative_theme,
    key_story_arc,
    content_flow,
    total_pieces,
    completed_pieces
)
SELECT 
    c.id as campaign_id,
    COALESCE(c.name, 'Campaign Content Strategy') as title,
    'Content narrative created from migrated campaign tasks' as description,
    'Expert insights and thought leadership' as narrative_theme,
    ARRAY['Industry expertise', 'Solution presentation', 'Practical implementation', 'Future outlook'] as key_story_arc,
    jsonb_build_object(
        'deliverables_order', deliverable_ids,
        'migration_source', 'campaign_tasks',
        'created_at', NOW()
    ) as content_flow,
    deliverable_count as total_pieces,
    approved_count as completed_pieces
FROM (
    SELECT 
        c.id,
        c.name,
        COUNT(cd.id) as deliverable_count,
        COUNT(CASE WHEN cd.status IN ('approved', 'published') THEN 1 END) as approved_count,
        array_agg(cd.id ORDER BY cd.narrative_order) as deliverable_ids
    FROM campaigns c
    INNER JOIN content_deliverables cd ON cd.campaign_id = c.id
    WHERE cd.metadata->>'migrated_from_task' = 'true'
    GROUP BY c.id, c.name
    HAVING COUNT(cd.id) > 0
) c
WHERE NOT EXISTS (
    SELECT 1 FROM content_narratives cn WHERE cn.campaign_id = c.id
);

-- Step 8: Log migration results
DO $$
DECLARE
    migrated_count INTEGER;
    narrative_count INTEGER;
    linked_count INTEGER;
BEGIN
    -- Count migrated deliverables
    SELECT COUNT(*) INTO migrated_count 
    FROM content_deliverables 
    WHERE metadata->>'migrated_from_task' = 'true';
    
    -- Count created narratives
    SELECT COUNT(*) INTO narrative_count 
    FROM content_narratives 
    WHERE content_flow->>'migration_source' = 'campaign_tasks';
    
    -- Count linked tasks
    SELECT COUNT(*) INTO linked_count 
    FROM campaign_tasks 
    WHERE deliverable_id IS NOT NULL;
    
    -- Log results
    RAISE NOTICE '=== TASK TO DELIVERABLE MIGRATION COMPLETED ===';
    RAISE NOTICE 'Migrated Deliverables: %', migrated_count;
    RAISE NOTICE 'Created Narratives: %', narrative_count;
    RAISE NOTICE 'Linked Tasks: %', linked_count;
    RAISE NOTICE '============================================';
END $$;

-- Step 9: Clean up functions (optional - comment out if you want to keep them)
-- DROP FUNCTION IF EXISTS map_task_type_to_content_type(TEXT);
-- DROP FUNCTION IF EXISTS map_task_status_to_deliverable_status(TEXT);
-- DROP FUNCTION IF EXISTS extract_title_from_result(TEXT, TEXT);
-- DROP FUNCTION IF EXISTS extract_key_messages_from_result(TEXT, TEXT);

-- Add helpful comments
COMMENT ON TABLE content_deliverables IS 'Content deliverables - migrated from campaign tasks. Focus on actual content rather than task execution.';
COMMENT ON COLUMN content_deliverables.metadata IS 'Includes migration info: original_task_id, migration_date, etc.';

-- Add validation constraints to ensure data quality
ALTER TABLE content_deliverables ADD CONSTRAINT check_title_length CHECK (char_length(title) >= 3);
ALTER TABLE content_deliverables ADD CONSTRAINT check_content_length CHECK (char_length(content) >= 10);