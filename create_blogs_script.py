#!/usr/bin/env python3
"""
Script para crear blogs automáticamente usando la API de Credilinq
"""

import requests
import time
import json

# URL de tu API (cambiar si es diferente)
API_BASE_URL = "https://credilinq-blog-production.up.railway.app"

# Contexto de la empresa (mismo que usa el frontend)
COMPANY_CONTEXT = """Credilinq.ai is a fintech leader in embedded lending and B2B credit solutions across Southeast Asia. We help businesses access funding through embedded financial products and innovative credit infrastructure."""

# Lista de todos los titulares organizados por categoría
BLOG_TITLES = [
    # Cash Flow & Working Capital Education
    "What Is Working Capital—and Why It's Critical for Marketplace Sellers",
    "The Cash Flow Gap Explained: How Payment Delays Hurt Growing Businesses",
    "5 Signs Your Business Has a Cash Flow Problem (And What to Do About It)",
    "How to Manage Cash Flow During Seasonal Highs and Lows",
    "Working Capital 101: A Beginner's Guide for B2B and Ecommerce Sellers",
    "How to Calculate Your Cash Flow Breakeven—and Why It Matters",
    "Cash Flow vs. Profit: Why Marketplace Sellers Must Know the Difference",
    "How to Use Working Capital to Fund Growth Without Taking on Long-Term Debt",
    "Why Fast Payouts Matter: The ROI of Closing the Cash Flow Gap",
    "3 Cash Flow Strategies for Surviving Net-30, Net-60, and Net-90 Payment Terms",
    
    # Financing Options & Embedded Lending
    "Traditional Loans vs. Embedded Lending: What's Better for Marketplace Sellers?",
    "What Is Embedded Lending? A Guide for B2B Platforms and Their Sellers",
    "The Rise of Platform-Based Credit: How Fintech Is Changing SME Financing",
    "Short-Term Loans for Digital Sellers: Pros, Cons, and Use Cases",
    "The Business Case for Offering Embedded Finance on Your B2B Platform",
    "How Embedded Lending Helps Platforms Increase Seller Retention and GMV",
    "How to Qualify for a CrediLinq Cash Advance (Even Without a Traditional Credit Score)",
    "The Future of SME Credit Is Embedded—Here's Why",
    "Is Invoice Factoring Right for You? Or Is Embedded Lending a Better Option?",
    "How to Access Working Capital Without Putting Up Personal Collateral",
    
    # Marketplace & Ecommerce Seller Insights
    "The Top 5 Cash Flow Challenges Facing Marketplace Sellers in 2025",
    "How Net Terms Hurt Marketplace Sellers—and What Platforms Can Do About It",
    "Cash Flow Tips for B2B Sellers on Amazon Business, Faire, and Other Platforms",
    "Why Digital Sellers Are Turned Down by Banks (and What to Do Instead)",
    "How to Use Financing to Scale Your Marketplace Store Faster",
    "The Best Times to Apply for Working Capital as an Online Seller",
    "What to Do When Your Platform Delays Your Payouts",
    "Inventory Financing vs. Cash Advance: Which Helps You Restock Faster?",
    "How Marketplace Sellers Can Use Working Capital to Fund Advertising Campaigns",
    "Lessons from Top-Performing Marketplace Sellers Who Use Embedded Finance",
    
    # Tactical, How-To, and Educational Content
    "How to Build a 12-Month Cash Flow Forecast (Template Included)",
    "5 Ways to Use a CrediLinq Cash Advance to Grow Your Business This Quarter",
    "How to Prepare Your Marketplace Store for Financing Approval",
    "The Smart Seller's Guide to Using Credit Without Overextending",
    "How to Talk to Your Accountant About Working Capital Loans",
    "How to Track and Improve Your Working Capital Cycle",
    "How to Use Financing Strategically During Slow Sales Months",
    "The Ultimate Glossary of Lending Terms Every SME Should Know",
    "How to Read Your CrediLinq Offer: APR, Repayments, and Terms Explained",
    "Using Capital to Go Multichannel: Financing Your Expansion into New Marketplaces",
    
    # Thought Leadership & Industry Trends
    "Why Cash Flow Is the #1 Barrier to SME Growth in the U.S.",
    "The Embedded Finance Revolution: What It Means for B2B Commerce",
    "How Fintech Is Closing the Credit Gap for Underserved Small Businesses",
    "The Future of SME Lending: Predictive Credit Models and Real-Time Risk",
    "What Online Platforms Can Learn from Fintech Lenders",
    "Why Speed Matters in SME Financing—and How Platforms Can Help",
    "Embedded Lending in Construction Marketplaces: A Game Changer for Merchants",
    "How Fintech Is Making Credit More Equitable for Digital-First Businesses",
    "Why Credit Invisibility Still Affects 27% of U.S. Small Businesses—and How to Fix It",
    "What Investors Are Watching in the Embedded Lending Space for 2025"
]

def create_blog(title, company_context=COMPANY_CONTEXT, content_type="blog"):
    """Crear un blog usando la API"""
    url = f"{API_BASE_URL}/blogs"
    
    payload = {
        "title": title,
        "company_context": company_context,
        "content_type": content_type
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating blog '{title}': {e}")
        return None

def main():
    """Función principal para crear todos los blogs"""
    print(f"🚀 Iniciando creación de {len(BLOG_TITLES)} blogs...")
    print(f"📡 API URL: {API_BASE_URL}")
    print("-" * 60)
    
    created_blogs = []
    failed_blogs = []
    
    for i, title in enumerate(BLOG_TITLES, 1):
        print(f"[{i}/{len(BLOG_TITLES)}] Creando: {title}")
        
        result = create_blog(title)
        
        if result:
            created_blogs.append(result)
            print(f"✅ Blog creado exitosamente - ID: {result.get('id', 'N/A')}")
        else:
            failed_blogs.append(title)
            print(f"❌ Falló la creación del blog")
        
        # Pausa entre requests para no sobrecargar la API
        if i < len(BLOG_TITLES):
            print("⏳ Esperando 2 segundos...")
            time.sleep(2)
        
        print("-" * 60)
    
    # Resumen final
    print("\n📊 RESUMEN FINAL:")
    print(f"✅ Blogs creados exitosamente: {len(created_blogs)}")
    print(f"❌ Blogs que fallaron: {len(failed_blogs)}")
    
    if failed_blogs:
        print("\n📝 Blogs que fallaron:")
        for title in failed_blogs:
            print(f"  - {title}")
    
    if created_blogs:
        print(f"\n🎉 ¡{len(created_blogs)} blogs creados exitosamente!")
        print("Puedes verlos en: https://credilinq-blog.vercel.app/")

if __name__ == "__main__":
    main() 