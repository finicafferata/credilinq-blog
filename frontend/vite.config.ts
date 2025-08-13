import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      treeshake: {
        moduleSideEffects: false,
        propertyReadSideEffects: false,
        tryCatchDeoptimization: false
      },
      output: {
        // Manual chunk splitting for better caching
        manualChunks: (id) => {
          // Vendor chunks - split into smaller chunks
          if (id.includes('node_modules')) {
            if (id.includes('react') && !id.includes('react-router') && !id.includes('react-hot-toast')) {
              return 'react-vendor';
            }
            if (id.includes('react-router')) {
              return 'router-vendor';
            }
            if (id.includes('react-hot-toast')) {
              return 'toast-vendor';
            }
            if (id.includes('axios')) {
              return 'axios-vendor';
            }
            if (id.includes('@heroicons')) {
              return 'heroicons-vendor';
            }
            // Other vendor libraries
            return 'vendor';
          }
          
          // Feature-based chunks with more granular splitting
          if (id.includes('/competitor-intelligence') || 
              id.includes('CompetitorIntelligence') || 
              id.includes('competitor-intelligence-api')) {
            return 'competitor-intelligence';
          }
          
          if (id.includes('/Analytics') || 
              id.includes('AnalyticsDashboard') || 
              id.includes('AdvancedReporting') ||
              id.includes('useAnalytics')) {
            return 'analytics';
          }
          
          if (id.includes('EnhancedBlog') || 
              id.includes('WorkflowPage') ||
              id.includes('BlogList') ||
              id.includes('BlogCard')) {
            return 'content-creation';
          }
          
          if (id.includes('Campaign') && !id.includes('BlogList')) {
            return 'campaigns';
          }
          
          if (id.includes('/Settings') || 
              id.includes('IntegrationManagement')) {
            return 'settings';
          }
          
          // Common components
          if (id.includes('/components/') && !id.includes('BlogList')) {
            return 'shared-components';
          }
        },
        // Optimize chunk file names for better caching
        chunkFileNames: () => {
          return `js/[name]-[hash].js`
        },
        entryFileNames: 'js/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.')
          const ext = info[info.length - 1]
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return `images/[name]-[hash][extname]`
          } else if (/woff2?|eot|ttf|otf/i.test(ext)) {
            return `fonts/[name]-[hash][extname]`
          } else {
            return `assets/[name]-[hash][extname]`
          }
        }
      }
    },
    // Optimize build
    target: 'esnext',
    minify: 'esbuild',
    cssCodeSplit: true,
    // Reduce bundle size
    chunkSizeWarningLimit: 400,
    // Performance optimizations
    reportCompressedSize: false,
    write: true
  },
  optimizeDeps: {
    include: [
      'react', 
      'react-dom', 
      'react-router-dom', 
      'axios',
      '@heroicons/react'
    ],
    exclude: ['@vite/client', '@vite/env']
  },
  // Performance optimizations
  esbuild: {
    // Remove console logs in production
    drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : []
  }
})
