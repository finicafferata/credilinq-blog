// Service Worker for CrediLinq AI Content Platform
// Provides offline functionality and intelligent caching

const CACHE_NAME = 'credilinq-v1.0.0'
const API_CACHE_NAME = 'credilinq-api-v1.0.0'

// Resources to cache immediately
const STATIC_RESOURCES = [
  '/',
  '/dashboard',
  '/static/js/index.js',
  '/static/css/index.css',
  '/favicon.ico'
]

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/health',
  '/api/v2/blogs',
  '/api/v2/campaigns',
  '/api/v2/analytics/dashboard'
]

// Install event - cache static resources
self.addEventListener('install', (event) => {
  console.log('Service Worker installing...')
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Caching static resources')
        return cache.addAll(STATIC_RESOURCES)
      })
      .catch((error) => {
        console.error('Failed to cache static resources:', error)
      })
  )
  
  // Skip waiting to activate new service worker immediately
  self.skipWaiting()
})

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...')
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
            console.log('Deleting old cache:', cacheName)
            return caches.delete(cacheName)
          }
        })
      )
    })
  )
  
  // Take control of all pages immediately
  self.clients.claim()
})

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const request = event.request
  const url = new URL(request.url)
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return
  }
  
  // Skip Chrome extensions and other non-http(s) requests
  if (!url.protocol.startsWith('http')) {
    return
  }
  
  event.respondWith(
    handleRequest(request)
  )
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  try {
    // Handle API requests
    if (url.pathname.startsWith('/api/')) {
      return handleAPIRequest(request)
    }
    
    // Handle static assets
    if (isStaticAsset(request)) {
      return handleStaticAsset(request)
    }
    
    // Handle navigation requests
    if (request.mode === 'navigate') {
      return handleNavigation(request)
    }
    
    // Default: network first
    return fetch(request)
    
  } catch (error) {
    console.error('Service Worker fetch error:', error)
    return new Response('Service Unavailable', { status: 503 })
  }
}

// API request handling - Network first with cache fallback
async function handleAPIRequest(request) {
  const url = new URL(request.url)
  
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    // Cache successful GET responses for specific endpoints
    if (networkResponse.ok && shouldCacheAPI(url.pathname)) {
      const cache = await caches.open(API_CACHE_NAME)
      
      // Clone response before caching (response can only be consumed once)
      const responseClone = networkResponse.clone()
      
      // Add cache headers for better management
      const responseWithHeaders = new Response(responseClone.body, {
        status: responseClone.status,
        statusText: responseClone.statusText,
        headers: {
          ...responseClone.headers,
          'sw-cached': 'true',
          'sw-cached-at': Date.now().toString()
        }
      })
      
      cache.put(request, responseWithHeaders)
    }
    
    return networkResponse
    
  } catch (error) {
    // Network failed, try cache
    console.log('API network failed, trying cache:', request.url)
    
    const cachedResponse = await caches.match(request)
    
    if (cachedResponse) {
      // Add header to indicate this is from cache
      const headers = new Headers(cachedResponse.headers)
      headers.set('sw-from-cache', 'true')
      
      return new Response(cachedResponse.body, {
        status: cachedResponse.status,
        statusText: cachedResponse.statusText,
        headers: headers
      })
    }
    
    // No cache available, return error response
    return new Response(
      JSON.stringify({ 
        error: 'Network unavailable and no cached data available',
        offline: true 
      }),
      { 
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

// Static asset handling - Cache first with network fallback
async function handleStaticAsset(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request)
    
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Not in cache, fetch from network
    const networkResponse = await fetch(request)
    
    if (networkResponse.ok) {
      // Cache the response
      const cache = await caches.open(CACHE_NAME)
      cache.put(request, networkResponse.clone())
    }
    
    return networkResponse
    
  } catch (error) {
    console.error('Static asset fetch failed:', error)
    
    // Return a fallback if available
    return new Response('Asset not available offline', { status: 404 })
  }
}

// Navigation handling - Network first with offline fallback
async function handleNavigation(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    if (networkResponse.ok) {
      return networkResponse
    }
    
    // Network returned error, try cache
    const cachedResponse = await caches.match(request)
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Return cached index.html for SPA routing
    return caches.match('/')
    
  } catch (error) {
    // Network failed, try cache
    const cachedResponse = await caches.match(request)
    
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Fallback to index.html for SPA
    const indexResponse = await caches.match('/')
    
    if (indexResponse) {
      return indexResponse
    }
    
    // Last resort - return offline page
    return new Response(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Offline - CrediLinq</title>
          <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .offline-container { max-width: 400px; margin: 0 auto; }
            .offline-icon { font-size: 64px; margin-bottom: 20px; }
          </style>
        </head>
        <body>
          <div class="offline-container">
            <div class="offline-icon">📡</div>
            <h1>You're Offline</h1>
            <p>Please check your internet connection and try again.</p>
            <button onclick="location.reload()">Retry</button>
          </div>
        </body>
      </html>
    `, {
      status: 200,
      headers: { 'Content-Type': 'text/html' }
    })
  }
}

// Utility functions
function isStaticAsset(request) {
  const url = new URL(request.url)
  return (
    url.pathname.startsWith('/static/') ||
    url.pathname.startsWith('/assets/') ||
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.jpg') ||
    url.pathname.endsWith('.jpeg') ||
    url.pathname.endsWith('.gif') ||
    url.pathname.endsWith('.svg') ||
    url.pathname.endsWith('.woff') ||
    url.pathname.endsWith('.woff2')
  )
}

function shouldCacheAPI(pathname) {
  // Cache read-only API endpoints
  const cacheablePatterns = [
    '/api/health',
    '/api/v2/blogs',
    '/api/v2/campaigns',
    '/api/v2/analytics',
    '/api/v2/competitor-intelligence/competitors',
    '/api/v2/competitor-intelligence/dashboard'
  ]
  
  return cacheablePatterns.some(pattern => 
    pathname.startsWith(pattern) && !pathname.includes('/analyze')
  )
}

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting()
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    const urls = event.data.urls
    event.waitUntil(
      caches.open(CACHE_NAME).then(cache => cache.addAll(urls))
    )
  }
})

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    console.log('Background sync triggered')
    event.waitUntil(handleBackgroundSync())
  }
})

async function handleBackgroundSync() {
  // Handle any queued actions when connection is restored
  console.log('Processing background sync...')
  
  // This could include:
  // - Sending queued blog posts
  // - Syncing analytics data
  // - Updating cached content
}

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'content-sync') {
    event.waitUntil(syncContent())
  }
})

async function syncContent() {
  console.log('Periodic content sync...')
  
  // Update cached content periodically
  try {
    const cache = await caches.open(API_CACHE_NAME)
    
    // Update dashboard data
    const dashboardResponse = await fetch('/api/v2/analytics/dashboard')
    if (dashboardResponse.ok) {
      cache.put('/api/v2/analytics/dashboard', dashboardResponse.clone())
    }
    
    // Update other critical data
    console.log('Content sync completed')
  } catch (error) {
    console.error('Content sync failed:', error)
  }
}

console.log('Service Worker loaded successfully')