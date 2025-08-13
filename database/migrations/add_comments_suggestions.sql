-- Comments & Suggestions schema for editor collaboration (snake_case aligned)

-- Blog comments
CREATE TABLE IF NOT EXISTS blog_comment (
  id UUID PRIMARY KEY,
  blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
  author TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved BOOLEAN NOT NULL DEFAULT FALSE,
  position_start INT,
  position_end INT,
  position_selected_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_blog_comment_blog_id ON blog_comment (blog_id);
CREATE INDEX IF NOT EXISTS idx_blog_comment_timestamp ON blog_comment (timestamp);

-- Replies to comments
CREATE TABLE IF NOT EXISTS blog_comment_reply (
  id UUID PRIMARY KEY,
  comment_id UUID NOT NULL REFERENCES blog_comment(id) ON DELETE CASCADE,
  author TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_blog_comment_reply_comment_id ON blog_comment_reply (comment_id);
CREATE INDEX IF NOT EXISTS idx_blog_comment_reply_timestamp ON blog_comment_reply (timestamp);

-- Suggestions
CREATE TABLE IF NOT EXISTS blog_suggestion (
  id UUID PRIMARY KEY,
  blog_id UUID NOT NULL REFERENCES blog_posts(id) ON DELETE CASCADE,
  author TEXT NOT NULL,
  original_text TEXT NOT NULL,
  suggested_text TEXT NOT NULL,
  reason TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','rejected')),
  position_start INT NOT NULL,
  position_end INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blog_suggestion_blog_id ON blog_suggestion (blog_id);
CREATE INDEX IF NOT EXISTS idx_blog_suggestion_status ON blog_suggestion (status);
CREATE INDEX IF NOT EXISTS idx_blog_suggestion_timestamp ON blog_suggestion (timestamp);

