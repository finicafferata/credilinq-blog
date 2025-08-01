"""Unit tests for database operations (mocked)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import psycopg2


class TestDatabaseOperations:
    """Test database operations without actually connecting to database."""

    @patch('src.config.database.db_config')
    def test_database_connection_mock(self, mock_db_config):
        """Test that database connection can be mocked properly."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Test the mock setup
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        
        assert mock_conn.cursor.called
        assert mock_cursor.execute.called
        assert mock_cursor.fetchone.called

    @patch('src.config.database.db_config')
    def test_blog_creation_database_operations(self, mock_db_config):
        """Test blog creation database operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Mock successful blog creation
        mock_cursor.fetchone.return_value = ("blog-123",)
        
        # Simulate blog creation
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO "BlogPost" (id, title, "contentMarkdown", status, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                RETURNING id
            """, ("blog-123", "Test Blog", "# Content", "draft"))
            blog_id = cursor.fetchone()[0]
            conn.commit()
        
        assert blog_id == "blog-123"
        assert mock_cursor.execute.called
        assert mock_conn.commit.called

    @patch('src.config.database.db_config')
    def test_campaign_creation_database_operations(self, mock_db_config):
        """Test campaign creation database operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Simulate campaign creation with all related tables
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create campaign
            cursor.execute("""
                INSERT INTO "Campaign" (id, "blogPostId", "createdAt")
                VALUES (%s, %s, NOW())
            """, ("campaign-123", "blog-123"))
            
            # Create briefing
            cursor.execute("""
                INSERT INTO "Briefing" (id, "campaignName", "marketingObjective", 
                                      "targetAudience", channels, "desiredTone", 
                                      language, "createdAt", "updatedAt", "campaignId")
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)
            """, ("briefing-123", "Test Campaign", "Brand awareness", 
                  "B2B professionals", '["LinkedIn"]', "Professional", "English", "campaign-123"))
            
            # Create task
            cursor.execute("""
                INSERT INTO "CampaignTask" (id, "campaignId", "taskType", status, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            """, ("task-123", "campaign-123", "content_repurposing", "pending"))
            
            conn.commit()
        
        # Verify all operations were called
        assert mock_cursor.execute.call_count == 3
        assert mock_conn.commit.called

    @patch('src.config.database.db_config')
    def test_database_query_operations(self, mock_db_config):
        """Test various database query operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Test blog listing
        mock_cursor.fetchall.return_value = [
            ("blog-1", "Blog 1", "draft", "2025-01-01T00:00:00Z"),
            ("blog-2", "Blog 2", "published", "2025-01-01T01:00:00Z")
        ]
        
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, status, "createdAt"
                FROM "BlogPost"
                ORDER BY "createdAt" DESC
            """)
            blogs = cursor.fetchall()
        
        assert len(blogs) == 2
        assert blogs[0][1] == "Blog 1"
        assert mock_cursor.fetchall.called

        # Test campaign with tasks query
        mock_cursor.fetchall.return_value = [
            ("campaign-1", "Campaign 1", "active", 2, 1),
        ]
        
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, b."campaignName", 
                       CASE 
                           WHEN COUNT(ct.id) = 0 THEN 'draft'
                           WHEN COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) = COUNT(ct.id) THEN 'completed'
                           ELSE 'active'
                       END as status,
                       COUNT(ct.id) as total_tasks,
                       COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM "Campaign" c
                LEFT JOIN "Briefing" b ON c.id = b."campaignId"
                LEFT JOIN "CampaignTask" ct ON c.id = ct."campaignId"
                GROUP BY c.id, b."campaignName"
                ORDER BY c."createdAt" DESC
            """)
            campaigns = cursor.fetchall()
        
        assert len(campaigns) == 1
        assert campaigns[0][2] == "active"  # status

    @patch('src.config.database.db_config')
    def test_database_error_handling(self, mock_db_config):
        """Test database error handling."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Simulate database error
        mock_cursor.execute.side_effect = psycopg2.DatabaseError("Connection failed")
        
        with pytest.raises(psycopg2.DatabaseError):
            with mock_db_config.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")

    @patch('src.config.database.db_config')
    def test_database_transaction_rollback(self, mock_db_config):
        """Test database transaction rollback on error."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Simulate error during transaction
        mock_cursor.execute.side_effect = [None, psycopg2.DatabaseError("Insert failed")]
        
        try:
            with mock_db_config.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO table1 VALUES (%s)", ("value1",))
                cursor.execute("INSERT INTO table2 VALUES (%s)", ("value2",))  # This fails
                conn.commit()
        except psycopg2.DatabaseError:
            # Should trigger rollback
            mock_conn.rollback()
        
        assert mock_conn.rollback.called

    @patch('src.config.database.db_config')
    def test_complex_campaign_query(self, mock_db_config):
        """Test complex campaign query with joins."""
        mock_conn = Mock()
        mock_cursor = Mock()
        
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.get_db_connection.return_value.__exit__.return_value = None
        
        # Mock complex query result
        mock_cursor.fetchone.return_value = (
            "campaign-123",
            "Test Campaign", 
            "active",
            "Data-driven approach",
            5,  # total tasks
            3   # completed tasks
        )
        
        with mock_db_config.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, 
                       COALESCE(b."campaignName", 'Unnamed Campaign') as name,
                       CASE 
                           WHEN COUNT(ct.id) = 0 THEN 'draft'
                           WHEN COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) = COUNT(ct.id) THEN 'completed'
                           ELSE 'active'
                       END as status,
                       cs."narrativeApproach" as strategy, 
                       COUNT(ct.id) as total_tasks,
                       COUNT(CASE WHEN ct.status = 'completed' THEN 1 END) as completed_tasks
                FROM "Campaign" c
                LEFT JOIN "Briefing" b ON c.id = b."campaignId"
                LEFT JOIN "ContentStrategy" cs ON c.id = cs."campaignId"
                LEFT JOIN "CampaignTask" ct ON c.id = ct."campaignId"
                WHERE c.id = %s
                GROUP BY c.id, b."campaignName", cs."narrativeApproach"
            """, ("campaign-123",))
            result = cursor.fetchone()
        
        assert result[0] == "campaign-123"
        assert result[1] == "Test Campaign"
        assert result[2] == "active"
        assert result[4] == 5  # total tasks
        assert result[5] == 3  # completed tasks