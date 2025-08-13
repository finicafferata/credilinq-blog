import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { register as registerSW, showUpdateNotification } from './utils/serviceWorker'

// Register service worker for offline functionality
if (import.meta.env.PROD) {
  registerSW({
    onUpdate: (registration) => {
      console.log('App update available!')
      showUpdateNotification(registration)
    },
    onSuccess: (registration) => {
      console.log('App cached for offline use')
    }
  })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
