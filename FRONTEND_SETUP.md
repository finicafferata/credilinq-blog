# 🎨 CrediLinQ Content Agent - Frontend

A modern and professional React application for managing AI-generated blogs with a multi-agent system.

## 🏗️ Frontend Architecture

### **Technology Stack**
- ⚛️ **React 18** with TypeScript
- ⚡ **Vite** as build tool (ultra-fast)
- 🎨 **Tailwind CSS** for modern styling
- 🛣️ **React Router DOM** for navigation
- 📡 **Axios** for API communication
- 🎯 **Responsive Design** for mobile/desktop

### **Project Structure**
```
frontend/
├── src/
│   ├── components/         # Reusable components
│   │   └── Header.tsx      # Main navigation
│   ├── pages/              # Main pages
│   │   ├── Dashboard.tsx   # Blog list
│   │   ├── NewBlog.tsx     # Blog creation
│   │   ├── BlogEditor.tsx  # AI-powered editor
│   │   └── KnowledgeBase.tsx # RAG management
│   ├── lib/
│   │   └── api.ts          # API client
│   ├── App.tsx             # Main router
│   └── index.css           # Global styles
├── package.json
├── vite.config.ts
└── tailwind.config.cjs
```

## 🖥️ Pages and Functionality

### **1. Dashboard (`/`)**
- 📋 **Blog List**: Complete overview of all generated blogs
- 🔍 **Search & Filter**: Find blogs by title or status
- 📊 **Status Indicators**: draft, published, edited
- ⚡ **Quick Actions**: Edit, delete, publish

### **2. Blog Creation (`/new`)**
- 📝 **Simple Form**: Title, company context, content type
- 🤖 **AI Generation**: Multi-agent workflow in real-time
- ⏱️ **Progress Indicator**: Visual feedback during generation
- 📖 **Instant Preview**: See result immediately

### **3. Blog Editor (`/edit/:id`)**
- ✏️ **Manual Editing**: Direct markdown editing
- 🔄 **AI Revision**: Intelligent content improvement
- 👁️ **Live Preview**: See changes in real-time
- 💾 **Auto-save**: Prevent content loss

### **4. Knowledge Base (`/knowledge-base`)**
- 📤 **Document Upload**: PDF, TXT, DOCX support
- 🔍 **Vector Search**: RAG-powered content discovery
- 📚 **Document Management**: Organize knowledge resources
- 🧠 **AI Context**: Enhance blog generation with custom data

### **5. Campaign View (`/campaign/:id`)**
- 📋 **Task Overview**: Content repurposing tasks
- 🎯 **Multi-format**: LinkedIn, Twitter, Instagram adaptations
- 🖼️ **Image Generation**: AI-powered visual content
- 📈 **Progress Tracking**: Task status and completion

## 🎨 Design System

### **Color Palette**
```css
/* Primary Colors */
--primary-blue: #2563eb
--primary-green: #059669
--primary-gray: #374151

/* Status Colors */
--status-draft: #f59e0b
--status-published: #10b981
--status-error: #ef4444
```

### **Typography**
- **Headings**: Inter font, clean and modern
- **Body**: System fonts for optimal readability
- **Code**: Monospace for technical content

### **Components**
- 🔘 **Buttons**: Primary, secondary, danger variants
- 📄 **Cards**: Consistent shadows and spacing
- 🚨 **Alerts**: Success, warning, error states
- 📊 **Loading**: Elegant spinners and skeletons

## 🔗 Backend Integration

### **API Configuration**
```typescript
// lib/api.ts
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for AI operations
  headers: {
    'Content-Type': 'application/json'
  }
})
```

### **Main Endpoints**
- **GET `/api/blogs`** → List all blogs  
- **POST `/api/blogs`** → Generate new blog with AI
- **GET `/api/blogs/:id`** → Get specific blog
- **PUT `/api/blogs/:id`** → Update blog content
- **DELETE `/api/blogs/:id`** → Soft delete blog
- **POST `/api/blogs/:id/publish`** → Publish blog

### **Real-time Features**
- ⏱️ **Progress Updates**: WebSocket-like experience with polling
- 🔄 **Auto-refresh**: Keep data synchronized
- 📡 **Error Handling**: Robust retry mechanisms

## 🚀 Development Commands

### **Basic Commands**
```bash
npm run dev         # Development server → http://localhost:5173
npm install         # Install dependencies
```

### **Build Commands**
```bash
npm run build       # Production build
npm run preview     # Preview build
npm run lint        # ESLint linting
```

### **Deployment**
```bash
npm run vercel-build  # Vercel-specific build script
```

## 🔧 Advanced Configuration

### **Environment Variables**
```bash
# In Vercel, configure:
VITE_API_BASE_URL=/api    # For production
```

### **Vite Configuration**
```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'  # Development proxy
    }
  }
})
```

### **Tailwind Setup**
```javascript
// tailwind.config.cjs
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#2563eb'
      }
    }
  }
}
```

### **💡 AI-Powered Revision Assistant**
```typescript
const handleRevision = async () => {
  setIsRevising(true)
  try {
    const response = await api.post(`/api/blogs/${blogId}/revise`, {
      content: currentContent,
      instructions: revisionInstructions
    })
    setContent(response.data.revisedContent)
  } catch (error) {
    showError('Revision failed')
  } finally {
    setIsRevising(false)
  }
}
```

## 🌟 Upcoming Improvements

- 🔄 **Real-time Collaboration**: Multiple editors
- 📊 **Advanced Analytics**: Content performance metrics
- 🎨 **Theme Customization**: Dark/light mode
- 📱 **Mobile App**: React Native version
- 🤖 **Smart Suggestions**: AI-powered writing assistance
- 🔗 **Social Integration**: Direct publishing to platforms

## 💻 How to Use the Application

1. **Start Backend**: `python -m src.main`
2. **Start Frontend**: `npm run dev`
3. **Create Blog**: Fill form on `/new`
4. **Wait for AI**: Multi-agent generation (~30-60s)
5. **Edit if Needed**: Use `/edit/:id` for adjustments
6. **Publish**: Click publish when ready

### **🔄 AI Revision Workflow**
1. **Select Content**: Highlight text to revise
2. **Add Instructions**: Tell AI what to improve
3. **Review Changes**: Compare before/after
4. **Accept/Reject**: Keep or discard revisions

This frontend provides an intuitive, professional interface for managing AI-generated content with real-time feedback and powerful editing capabilities.