#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

try:
    from src.config.database import db_config
    
    def check_blogs():
        """Check blogs in database for duplicates and issues"""
        try:
            with db_config.get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute('SELECT id, title, status, "createdAt" FROM "BlogPost" WHERE status != \'deleted\' ORDER BY "createdAt" DESC')
                rows = cur.fetchall()
                
                print(f'Total blogs: {len(rows)}')
                
                # Check for duplicates
                ids = [row[0] for row in rows if row[0]]
                unique_ids = set(ids)
                print(f'Unique IDs: {len(unique_ids)}')
                print(f'Duplicate IDs: {len(ids) - len(unique_ids)}')
                
                # Show first few blogs
                print('\nFirst 5 blogs:')
                for i, row in enumerate(rows[:5]):
                    print(f'{i+1}. ID: {row[0]}, Title: {row[1]}, Status: {row[2]}, Date: {row[3]}')
                    
                # Check for null IDs
                null_ids = [row for row in rows if not row[0]]
                print(f'\nBlogs with null ID: {len(null_ids)}')
                
        except Exception as e:
            print(f'Database error: {e}')
    
    if __name__ == "__main__":
        check_blogs()
        
except Exception as e:
    print(f'Import error: {e}') 