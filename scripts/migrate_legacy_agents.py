#!/usr/bin/env python3
"""
Automated Legacy Agent Migration Tool

This script helps developers migrate from legacy agents to LangGraph agents
by analyzing code, suggesting changes, and optionally applying automatic fixes.
"""

import os
import re
import ast
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import subprocess

@dataclass
class MigrationIssue:
    """Represents a legacy agent usage that needs migration"""
    file_path: str
    line_number: int
    line_content: str
    legacy_agent: str
    suggested_replacement: str
    migration_type: str
    confidence: str  # 'high', 'medium', 'low'

class LegacyAgentMigrator:
    """Tool for migrating legacy agents to LangGraph adapters"""
    
    # Legacy agent mappings to their LangGraph equivalents
    AGENT_MIGRATIONS = {
        'QualityReviewAgent': {
            'replacement': 'AdapterFactory.create_editor_adapter()',
            'import_old': 'from src.agents.specialized.quality_review_agent import QualityReviewAgent',
            'import_new': 'from src.agents.adapters.langgraph_legacy_adapter import AdapterFactory',
            'guide_url': 'docs/migration/quality-review-migration.md'
        },
        'BrandReviewAgent': {
            'replacement': 'AdapterFactory.create_brand_review_adapter()',
            'import_old': 'from src.agents.specialized.brand_review_agent import BrandReviewAgent',
            'import_new': 'from src.agents.adapters.langgraph_legacy_adapter import AdapterFactory',
            'guide_url': 'docs/migration/brand-review-migration.md'
        },
        'ContentQualityAgent': {
            'replacement': 'AdapterFactory.create_editor_adapter()',
            'import_old': 'from src.agents.specialized.content_quality_agent import ContentQualityAgent',
            'import_new': 'from src.agents.adapters.langgraph_legacy_adapter import AdapterFactory',
            'guide_url': 'docs/migration/content-quality-migration.md'
        },
        'FinalApprovalAgent': {
            'replacement': 'AdapterFactory.create_editor_adapter()',
            'import_old': 'from src.agents.specialized.final_approval_agent import FinalApprovalAgent',
            'import_new': 'from src.agents.adapters.langgraph_legacy_adapter import AdapterFactory',
            'guide_url': 'docs/migration/final-approval-migration.md'
        }
    }
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root).resolve()
        self.issues: List[MigrationIssue] = []
        
    def scan_project(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> List[MigrationIssue]:
        """Scan the entire project for legacy agent usage"""
        if include_patterns is None:
            include_patterns = ['**/*.py']
        if exclude_patterns is None:
            exclude_patterns = ['**/migrations/**', '**/tests/**', '**/venv/**', '**/__pycache__/**']
            
        self.issues = []
        
        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if self._should_exclude_file(file_path, exclude_patterns):
                    continue
                    
                self._scan_file(file_path)
        
        return self.issues
    
    def _should_exclude_file(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded from scanning"""
        relative_path = file_path.relative_to(self.project_root)
        
        for pattern in exclude_patterns:
            if relative_path.match(pattern):
                return True
        return False
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan a single file for legacy agent usage"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Look for imports
            self._scan_imports(file_path, lines)
            
            # Look for instantiations
            self._scan_instantiations(file_path, lines)
            
            # Look for type annotations
            self._scan_type_annotations(file_path, lines)
            
        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")
    
    def _scan_imports(self, file_path: Path, lines: List[str]) -> None:
        """Scan for legacy agent imports"""
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            for agent_name, migration in self.AGENT_MIGRATIONS.items():
                # Look for direct imports
                if f'from src.agents.specialized.{agent_name.lower().replace("agent", "_agent")} import {agent_name}' in line_stripped:
                    self.issues.append(MigrationIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=i,
                        line_content=line,
                        legacy_agent=agent_name,
                        suggested_replacement=migration['import_new'],
                        migration_type='import',
                        confidence='high'
                    ))
                
                # Look for alternative import styles
                elif f'import {agent_name}' in line_stripped and 'specialized' in line_stripped:
                    self.issues.append(MigrationIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=i,
                        line_content=line,
                        legacy_agent=agent_name,
                        suggested_replacement=migration['import_new'],
                        migration_type='import_alternative',
                        confidence='medium'
                    ))
    
    def _scan_instantiations(self, file_path: Path, lines: List[str]) -> None:
        """Scan for legacy agent instantiations"""
        for i, line in enumerate(lines, 1):
            for agent_name, migration in self.AGENT_MIGRATIONS.items():
                # Look for direct instantiation
                if f'{agent_name}()' in line:
                    self.issues.append(MigrationIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=i,
                        line_content=line,
                        legacy_agent=agent_name,
                        suggested_replacement=migration['replacement'],
                        migration_type='instantiation',
                        confidence='high'
                    ))
                
                # Look for instantiation with parameters
                elif f'{agent_name}(' in line and not f'def {agent_name}(' in line:
                    self.issues.append(MigrationIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=i,
                        line_content=line,
                        legacy_agent=agent_name,
                        suggested_replacement=migration['replacement'],
                        migration_type='instantiation_with_args',
                        confidence='medium'
                    ))
    
    def _scan_type_annotations(self, file_path: Path, lines: List[str]) -> None:
        """Scan for legacy agent type annotations"""
        for i, line in enumerate(lines, 1):
            for agent_name in self.AGENT_MIGRATIONS.keys():
                # Look for type annotations
                if f': {agent_name}' in line or f'-> {agent_name}' in line:
                    self.issues.append(MigrationIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=i,
                        line_content=line,
                        legacy_agent=agent_name,
                        suggested_replacement=f"# TODO: Update type annotation for {agent_name}",
                        migration_type='type_annotation',
                        confidence='low'
                    ))
    
    def generate_report(self, output_format: str = 'console') -> str:
        """Generate a migration report"""
        if output_format == 'console':
            return self._generate_console_report()
        elif output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'json':
            return self._generate_json_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_console_report(self) -> str:
        """Generate a console-friendly report"""
        report = []
        report.append("🚨 Legacy Agent Migration Report")
        report.append("=" * 50)
        report.append(f"Found {len(self.issues)} migration issues:\n")
        
        # Group by file
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        for file_path, file_issues in issues_by_file.items():
            report.append(f"📁 {file_path}")
            for issue in file_issues:
                confidence_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                report.append(f"  {confidence_emoji[issue.confidence]} Line {issue.line_number}: {issue.migration_type}")
                report.append(f"     Current: {issue.line_content.strip()}")
                report.append(f"     Replace: {issue.suggested_replacement}")
                
                # Add migration guide reference
                if issue.legacy_agent in self.AGENT_MIGRATIONS:
                    guide_url = self.AGENT_MIGRATIONS[issue.legacy_agent]['guide_url']
                    report.append(f"     Guide: {guide_url}")
                report.append("")
        
        # Summary
        report.append("\n📊 Summary:")
        report.append(f"   Total Issues: {len(self.issues)}")
        
        by_type = {}
        by_confidence = {}
        for issue in self.issues:
            by_type[issue.migration_type] = by_type.get(issue.migration_type, 0) + 1
            by_confidence[issue.confidence] = by_confidence.get(issue.confidence, 0) + 1
        
        report.append("   By Type:")
        for migration_type, count in by_type.items():
            report.append(f"     - {migration_type}: {count}")
        
        report.append("   By Confidence:")
        for confidence, count in by_confidence.items():
            report.append(f"     - {confidence}: {count}")
        
        return "\n".join(report)
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report"""
        report = []
        report.append("# 🚨 Legacy Agent Migration Report\n")
        report.append(f"Found **{len(self.issues)}** migration issues that need attention.\n")
        
        # Summary table
        report.append("## 📊 Summary\n")
        report.append("| Metric | Count |")
        report.append("|--------|-------|")
        report.append(f"| Total Issues | {len(self.issues)} |")
        
        by_confidence = {}
        for issue in self.issues:
            by_confidence[issue.confidence] = by_confidence.get(issue.confidence, 0) + 1
        
        for confidence, count in by_confidence.items():
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[confidence]
            report.append(f"| {emoji} {confidence.title()} Priority | {count} |")
        
        report.append("\n## 📁 Issues by File\n")
        
        # Group by file
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        for file_path, file_issues in sorted(issues_by_file.items()):
            report.append(f"### `{file_path}`\n")
            
            for issue in file_issues:
                confidence_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[issue.confidence]
                report.append(f"**{confidence_emoji} Line {issue.line_number}** - {issue.migration_type}")
                report.append("```python")
                report.append(f"# Current:")
                report.append(issue.line_content)
                report.append(f"# Replace with:")
                report.append(issue.suggested_replacement)
                report.append("```")
                
                if issue.legacy_agent in self.AGENT_MIGRATIONS:
                    guide_url = self.AGENT_MIGRATIONS[issue.legacy_agent]['guide_url']
                    report.append(f"📖 [Migration Guide]({guide_url})")
                
                report.append("")
        
        return "\n".join(report)
    
    def apply_automatic_fixes(self, dry_run: bool = True, confidence_threshold: str = 'high') -> Dict[str, int]:
        """Apply automatic fixes for high-confidence issues"""
        confidence_levels = {'high': 3, 'medium': 2, 'low': 1}
        min_confidence = confidence_levels[confidence_threshold]
        
        fixes_applied = {'files_modified': 0, 'issues_fixed': 0, 'issues_skipped': 0}
        
        # Group issues by file
        issues_by_file = {}
        for issue in self.issues:
            if confidence_levels[issue.confidence] >= min_confidence:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
            else:
                fixes_applied['issues_skipped'] += 1
        
        for file_path, file_issues in issues_by_file.items():
            if self._apply_file_fixes(file_path, file_issues, dry_run):
                fixes_applied['files_modified'] += 1
                fixes_applied['issues_fixed'] += len(file_issues)
        
        return fixes_applied
    
    def _apply_file_fixes(self, file_path: str, issues: List[MigrationIssue], dry_run: bool) -> bool:
        """Apply fixes to a single file"""
        full_path = self.project_root / file_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort issues by line number in reverse order to avoid offset issues
            issues.sort(key=lambda x: x.line_number, reverse=True)
            
            modified = False
            for issue in issues:
                if issue.migration_type in ['import', 'instantiation']:
                    # Simple string replacement
                    old_line = lines[issue.line_number - 1]
                    
                    if issue.migration_type == 'import':
                        new_line = issue.suggested_replacement + '\n'
                    else:  # instantiation
                        new_line = old_line.replace(f'{issue.legacy_agent}()', issue.suggested_replacement)
                    
                    lines[issue.line_number - 1] = new_line
                    modified = True
            
            if modified and not dry_run:
                # Write back to file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print(f"✅ Applied fixes to {file_path}")
            elif modified and dry_run:
                print(f"🔍 Would apply fixes to {file_path} (dry run)")
            
            return modified
            
        except Exception as e:
            print(f"❌ Error applying fixes to {file_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Legacy Agent Migration Tool')
    parser.add_argument('--project-root', '-r', default='.', help='Project root directory')
    parser.add_argument('--format', '-f', choices=['console', 'markdown', 'json'], default='console',
                       help='Report output format')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--fix', action='store_true', help='Apply automatic fixes')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without applying')
    parser.add_argument('--confidence', '-c', choices=['high', 'medium', 'low'], default='high',
                       help='Minimum confidence level for automatic fixes')
    
    args = parser.parse_args()
    
    migrator = LegacyAgentMigrator(args.project_root)
    
    print("🔍 Scanning project for legacy agent usage...")
    issues = migrator.scan_project()
    
    if not issues:
        print("✅ No legacy agent issues found!")
        return
    
    # Generate report
    report = migrator.generate_report(args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report written to {args.output}")
    else:
        print(report)
    
    # Apply fixes if requested
    if args.fix or args.dry_run:
        print(f"\n🔧 {'Applying' if args.fix else 'Previewing'} automatic fixes...")
        results = migrator.apply_automatic_fixes(
            dry_run=args.dry_run or not args.fix,
            confidence_threshold=args.confidence
        )
        
        print(f"📊 Fix Results:")
        print(f"   Files modified: {results['files_modified']}")
        print(f"   Issues fixed: {results['issues_fixed']}")
        print(f"   Issues skipped: {results['issues_skipped']}")
        
        if results['issues_fixed'] > 0 and not args.dry_run:
            print(f"\n✅ Applied {results['issues_fixed']} automatic fixes!")
            print("🔍 Please review changes and run tests to ensure correctness.")
    
    print(f"\n📖 Migration guides available:")
    for agent_name, migration in migrator.AGENT_MIGRATIONS.items():
        print(f"   - {agent_name}: {migration['guide_url']}")

if __name__ == '__main__':
    main()