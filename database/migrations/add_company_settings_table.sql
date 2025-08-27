-- Create company_settings table for storing company profile and configuration
CREATE TABLE IF NOT EXISTS company_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255),
    company_context TEXT,
    brand_voice TEXT,
    value_proposition TEXT,
    industries TEXT[], -- Array of industries
    target_audiences TEXT[], -- Array of target audiences
    tone_presets TEXT[], -- Array of tone presets
    keywords TEXT[], -- Array of keywords
    style_guidelines TEXT,
    prohibited_topics TEXT[], -- Array of prohibited topics
    compliance_notes TEXT,
    links JSONB, -- JSON array of {label, url} objects
    default_cta TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_company_settings_updated_at 
    BEFORE UPDATE ON company_settings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default row (there should only be one row for company settings)
INSERT INTO company_settings (id, company_name, company_context) 
VALUES ('00000000-0000-0000-0000-000000000001', 'Your Company', '')
ON CONFLICT (id) DO NOTHING;