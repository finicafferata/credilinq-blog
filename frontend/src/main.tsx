import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
// Service worker temporarily disabled for debugging
// import { register as registerSW, showUpdateNotification } from './utils/serviceWorker'

// Register service worker for offline functionality
// if (import.meta.env.PROD) {
//   registerSW({
//     onUpdate: (registration) => {
//       console.log('App update available!')
//       showUpdateNotification(registration)
//     },
//     onSuccess: (registration) => {
//       console.log('App cached for offline use')
//     }
//   })
// }

try {
  const rootElement = document.getElementById('root');
  if (!rootElement) {
    throw new Error('Root element not found');
  }
  
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  console.log('React app rendered successfully');
} catch (error) {
  console.error('Failed to render React app:', error);
  // Fallback: show error on page
  document.body.innerHTML = `
    <div style="padding: 20px; font-family: Arial, sans-serif;">
      <h1>Application Error</h1>
      <p>Failed to load the application: ${error}</p>
      <p>Please check the browser console for more details.</p>
    </div>
  `;
}
