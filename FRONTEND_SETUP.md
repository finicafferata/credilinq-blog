# 🎨 Credilinq Content Agent - Frontend

Una aplicación React moderna y profesional para gestionar blogs generados por IA con un sistema multi-agente.

## 🏗️ Arquitectura del Frontend

### **Stack Tecnológico**
- ⚛️ **React 18** con TypeScript
- ⚡ **Vite** como build tool (ultra-rápido)
- 🎨 **Tailwind CSS** para styling moderno
- 🛣️ **React Router DOM** para navegación
- 📡 **Axios** para comunicación con API
- 🎯 **Responsive Design** para móvil/desktop

### **Estructura del Proyecto**
```
frontend/
├── src/
│   ├── components/         # Componentes reutilizables
│   │   └── Header.tsx     # Navegación principal
│   ├── pages/             # Páginas principales
│   │   ├── Dashboard.tsx  # Lista de blogs
│   │   ├── NewBlog.tsx    # Creación de blogs
│   │   ├── BlogEditor.tsx # Editor con IA
│   │   └── KnowledgeBase.tsx # Gestión RAG
│   ├── lib/
│   │   └── api.ts         # Cliente API
│   ├── App.tsx            # Router principal
│   └── index.css          # Estilos globales
├── package.json
├── vite.config.ts
├── tailwind.config.cjs
└── postcss.config.cjs
```

---

## 🖥️ Páginas y Funcionalidades

### **1. Dashboard (`/`)**
- **📋 Lista de blogs** con cards visuales
- **🏷️ Estados**: draft, edited, published
- **📅 Fechas** de creación formateadas
- **⚡ Acciones**: Edit, Delete por blog
- **➕ Creación rápida** con botón prominente

### **2. Creación de Blogs (`/new`)**
- **📝 Formulario intuitivo** con:
  - Campo de título del blog
  - Área de contexto empresarial (pre-llenado con Credilinq.ai)
- **🤖 Explicación del proceso** con los 4 agentes
- **⏱️ Estados de carga** durante generación
- **🔄 Redirección automática** al editor tras creación

### **3. Editor de Blogs (`/edit/:blogId`)**
- **📝 Editor de markdown** completo (textarea avanzado)
- **✨ Asistente de revisión con IA**:
  - Selecciona texto → aparece botón "Revise Selected Text"
  - Modal con campo de instrucción ("Make this more concise")
  - Preview de cambios lado a lado (original vs. revisado)
  - Botones "Accept" / "Reject" para aplicar cambios
- **💾 Guardado** manual y automático
- **🔙 Navegación** de regreso al dashboard

### **4. Knowledge Base (`/knowledge-base`)**
- **📁 Drag & Drop** para subir documentos
- **📊 Progreso visual** de procesamiento
- **✅ Estados**: uploading, success, error
- **📋 Guía de mejores prácticas**
- **❓ Explicación del RAG** paso a paso

---

## 🎨 Sistema de Diseño

### **Colores (Credilinq.ai Theme)**
```css
Primary: #2563eb (blue-600)
Secondary: #f3f4f6 (gray-100)
Success: #10b981 (green-500)
Warning: #f59e0b (amber-500)
Error: #ef4444 (red-500)
```

### **Componentes CSS Personalizados**
- **`.btn-primary`**: Botones de acción principales
- **`.btn-secondary`**: Botones secundarios
- **`.card`**: Contenedores con sombra y border
- **`.input`**: Campos de entrada estilizados
- **`.textarea`**: Áreas de texto expandidas

### **Features UX/UI**
- **🎯 Hover effects** en botones y cards
- **⚡ Loading states** con spinners animados
- **📱 Responsive design** para móviles
- **🎨 Iconografía SVG** moderna
- **⌨️ Focus states** para accesibilidad

---

## 🔗 Integración con Backend

### **Configuración de API**
```typescript
// src/lib/api.ts
const api = axios.create({
  baseURL: isDev ? 'http://localhost:8000' : '/api',
});
```

### **Endpoints Consumidos**
- **POST `/blogs`** → Crear nuevo blog
- **GET `/blogs`** → Listar todos los blogs  
- **GET `/blogs/:id`** → Obtener blog específico
- **PUT `/blogs/:id`** → Actualizar contenido
- **POST `/blogs/:id/revise`** → Revisión con IA
- **POST `/documents/upload`** → Subir documentos RAG

### **Proxy de Desarrollo**
```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}
```

---

## 🚀 Comandos de Desarrollo

### **Desarrollo Local**
```bash
cd frontend
npm run dev         # Servidor desarrollo → http://localhost:5173
```

### **Build y Deploy**
```bash
npm run build       # Build para producción
npm run preview     # Preview del build
npm run lint        # Linting con ESLint
```

### **Vercel Deploy**
```bash
npm run vercel-build  # Script específico para Vercel
```

---

## 🔧 Configuración Avanzada

### **Variables de Entorno**
```env
# En Vercel, configurar:
VITE_API_BASE_URL=/api    # Para producción
```

### **Tailwind Config**
```javascript
// tailwind.config.cjs
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: { primary: {...}, gray: {...} },
      fontFamily: { sans: ['Inter', 'system-ui'] },
    },
  },
}
```

---

## 🎯 Funcionalidades Clave

### **💡 Asistente de Revisión con IA**
1. Usuario selecciona texto en el editor
2. Aparece botón flotante "✨ Revise Selected Text"
3. Modal con campo: "How should I revise this?"
4. AI procesa y muestra texto original vs. revisado
5. Usuario acepta o rechaza la sugerencia

### **📁 Sistema RAG Visual**
1. Drag & drop de archivos .txt/.md
2. Barra de progreso durante procesamiento
3. Estados visuales: uploading → processing → success
4. Explicación educativa del proceso RAG

### **🎨 Dashboard Intuitivo**
1. Cards organizadas en grid responsive
2. Estados con colores (draft=amarillo, edited=azul)
3. Fechas formateadas legibles
4. Acciones rápidas por blog

---

## 🌟 Próximas Mejoras

- [ ] **🔍 Búsqueda y filtrado** de blogs
- [ ] **📊 Analytics** de performance
- [ ] **🎨 Editor WYSIWYG** avanzado (TipTap/Quill)
- [ ] **🔄 Auto-save** cada 30 segundos
- [ ] **👥 Colaboración** multi-usuario
- [ ] **📱 PWA** para uso offline
- [ ] **🎯 Plantillas** de blogs predefinidas

---

## 💻 Cómo Usar la Aplicación

### **🎬 Flujo Completo**
1. **Dashboard** → Ver blogs existentes
2. **"+ Create New Blog"** → Formulario de creación
3. **AI Generation** → Los 4 agentes trabajan automáticamente
4. **Editor** → Seleccionar texto + revisar con IA
5. **Save** → Blog listo para publicación

### **🔄 Flujo de Revisión IA**
1. Seleccionar texto problemático
2. Click en "✨ Revise Selected Text"
3. Escribir instrucción: _"Make this more technical"_
4. Revisar sugerencia lado a lado
5. Accept → texto actualizado automáticamente

---

**🎉 ¡Tu aplicación Credilinq Content Agent está lista!**

Navega a **http://localhost:5173** para empezar a crear blogs con IA. 