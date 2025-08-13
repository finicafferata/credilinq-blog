# Analytics Dashboard - Calculation Methods

This document explains how each metric in the CrediLinq Analytics Dashboard is calculated and what it represents.

## 📊 Overview Dashboard Metrics

### Total Blogs
- **Calculation**: Count of all blog posts created in the system
- **Includes**: Published, draft, and archived posts across all campaigns
- **Example**: 12 total blog posts in the content library

### Total Campaigns  
- **Calculation**: Count of all marketing campaigns created
- **Includes**: Active, completed, and draft campaigns with associated tasks
- **Example**: 8 multi-channel content campaigns with goals and deliverables

### Agent Executions
- **Calculation**: Total number of AI agent tasks executed across all agent types
- **Includes**: All agent runs regardless of duration or success status
- **Agent Types**: Planner, Researcher, Writer, Editor, SEO, Image Prompt Generator
- **Example**: 156 total agent task executions

### Success Rate
- **Formula**: `(Successful Executions ÷ Total Executions) × 100`
- **Calculation**: Percentage of successful agent executions out of total executions
- **Example**: 87% = 135 successful tasks out of 156 total executions

## 📈 Performance Trends

### Daily Execution Count
- **Calculation**: Number of agent tasks completed each day
- **Tracking**: 7-day rolling window showing daily activity levels
- **Example**: Aug 13: 12 executions completed

### Daily Success Rate  
- **Formula**: `(Daily Successful Tasks ÷ Daily Total Tasks) × 100`
- **Calculation**: Percentage of successful tasks per day
- **Example**: Aug 13: 86.6% success rate (10-11 tasks succeeded out of 12)

## 🤖 Agent Performance Analytics

### Agent Execution Count
- **Calculation**: Number of tasks executed by each AI agent type during selected period
- **Breakdown by Agent Type**:
  - **Planner**: Content strategy and planning tasks
  - **Researcher**: Information gathering and analysis  
  - **Writer**: Content creation and drafting
  - **Editor**: Content review and improvement
  - **SEO**: Search engine optimization analysis
  - **Image Prompt Generator**: Visual content prompt creation

### Agent Success Rate
- **Formula**: `(Agent Successful Tasks ÷ Agent Total Tasks) × 100`
- **Calculation**: Success percentage for each individual agent type
- **Ranking**: Agents ranked by success rate with execution count as secondary factor

## 💰 Cost Distribution Analytics

### AI API Cost Calculation
- **Formula**: `Input Tokens × Input Rate + Output Tokens × Output Rate`
- **Token Rates**: Typically $0.001-$0.003 per 1000 tokens (varies by model)
- **Accumulation**: Costs sum across all executions per agent type
- **Total Cost Formula**: `Σ(Input Tokens × Input Rate + Output Tokens × Output Rate)` for each agent
- **Example**: Writer agent: $0.045 total cost from processing 25,000 tokens

### Performance Metrics Detail

#### Duration
- **Measurement**: Task execution time in seconds
- **Range**: Typically 1-30 seconds per task
- **Example**: 12.8s for a writing task

#### Quality Score  
- **Scale**: 0-100% performance quality rating
- **Calculation**: AI-generated quality assessment based on output analysis
- **Color Coding**: 
  - Green (80%+): High quality
  - Yellow (60-79%): Medium quality  
  - Red (<60%): Needs improvement

#### Token Usage
- **Format**: Input→Output token count
- **Calculation**: Tracks AI model token consumption
- **Example**: 186→293 tokens (186 input, 293 output, 479 total)

#### Cost per Execution
- **Formula**: `(Input + Output Tokens) ÷ 1000 × Token Rate`
- **Display**: USD cost per individual task execution
- **Example**: $0.0008 for a 479-token task

## 📝 Blog Performance Analytics

### Views
- **Calculation**: Total page views and unique visitors to blog posts
- **Tracking**: Cumulative view count across all traffic sources
- **Example**: 2,539 views for "AI Revolution in Content Marketing"

### Engagement Rate
- **Formula**: `(Interactions ÷ Total Views) × 100`
- **Interactions Include**: Comments, shares, time on page, scroll depth, click-throughs
- **Calculation**: Percentage of visitors who meaningfully interact with content
- **Example**: 5.04% engagement = ~128 interactions from 2,539 views

### Trend Indicators
- **Calculation**: Performance direction based on recent activity compared to historical data
- **Indicators**: 
  - ↗️ Trending Up: Increasing engagement/views
  - ↘️ Trending Down: Decreasing performance

## 🔍 Competitive Intelligence Analytics

### Total Competitors Monitored
- **Calculation**: Count of competitor companies actively tracked
- **Example**: 12 companies including TechCorp, InnovatePlus, DigitalEdge

### Active Monitoring
- **Calculation**: Number of competitors with recent activity (last 7 days)
- **Example**: 8 out of 12 competitors published content recently

### Content Analyzed  
- **Calculation**: Total competitor content pieces processed by AI algorithms
- **Content Types**: Blog posts, social media, press releases, videos
- **Example**: 347 pieces analyzed in 30 days

### Trends Identified
- **Calculation**: Number of emerging topics and patterns detected through content analysis
- **Method**: AI-powered sentiment tracking and topic modeling
- **Example**: 23 trends identified including "AI & Machine Learning" (+28% growth)

### Alerts Generated
- **Calculation**: Count of significant competitor activities triggering automated alerts
- **Trigger Events**: Major product launches, content campaigns, market positioning changes
- **Example**: 5 alerts for significant competitive activities

### Content Types Distribution
- **Calculation**: Percentage breakdown of competitor content by type
- **Categories**: Blog Posts, Social Media, Press Releases, Videos
- **Method**: Automated content classification and counting

### Platform Activity
- **Calculation**: Number of posts per social media platform
- **Platforms**: LinkedIn, Twitter, Facebook, Instagram, YouTube
- **Tracking**: Posts published during selected time period

### Top Competitors
- **Ranking**: Based on content volume and engagement metrics
- **Metrics**: Total content count, engagement rates, market presence
- **Display**: Company name, domain, content volume

### Trending Topics
- **Calculation**: Topic popularity with growth rate analysis
- **Formula**: `((Current Period Mentions - Previous Period) ÷ Previous Period) × 100`
- **Growth Rate**: Percentage change in topic mentions over time
- **Example**: "AI & Machine Learning": 162 mentions, +27.9% growth

## 🔄 Real-time Updates

### Auto-refresh
- **Frequency**: Configurable (10s, 30s, 1m, or manual)
- **Scope**: All dashboard metrics update simultaneously
- **Last Updated**: Timestamp showing most recent data refresh

### Data Freshness
- **Mock Data**: Currently using simulated data for demonstration
- **Production**: Would connect to live database with real-time metrics
- **Time Periods**: Selectable ranges (7d, 30d, 90d)

## 🛠️ Implementation Notes

### Current Status
- **Environment**: Local PostgreSQL with mock data for demonstration
- **Data Source**: Algorithmically generated realistic sample data
- **Purpose**: Showcase analytics capabilities and calculation methods

### Production Considerations
- **Real Data**: Would integrate with actual blog posts, campaigns, and agent executions
- **Performance**: Optimized queries and caching for large datasets
- **Scalability**: Designed to handle enterprise-level content analytics

---

*This documentation covers all calculation methods used in the CrediLinq Analytics Dashboard. Each metric includes hover tooltips in the UI for quick reference.*