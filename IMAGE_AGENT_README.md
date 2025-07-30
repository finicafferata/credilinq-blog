# Agente de Imágenes - CrediLinQ

## 🎯 Descripción

El Agente de Imágenes es un componente especializado que genera imágenes profesionales para contenido de blogs utilizando inteligencia artificial. Se integra perfectamente con el workflow de creación de contenido y proporciona una interfaz intuitiva para la generación y gestión de imágenes.

## ✨ Características Principales

### 🖼️ Generación de Imágenes
- **Generación automática** basada en el contenido del blog
- **Múltiples estilos** disponibles (Profesional, Creativo, Minimalista, Moderno, Vintage)
- **Configuración flexible** de cantidad de imágenes
- **Prompts inteligentes** generados automáticamente

### 🎨 Estilos Disponibles
- **Profesional**: Diseño limpio y corporativo
- **Creativo**: Estilo artístico y llamativo
- **Minimalista**: Diseño simple y elegante
- **Moderno**: Tendencias actuales de diseño
- **Vintage**: Estilo retro y clásico

### 🔄 Funcionalidades Avanzadas
- **Regeneración de imágenes** específicas
- **Vista previa** en modal
- **Descarga directa** de imágenes
- **Galería organizada** de imágenes generadas
- **Estadísticas** de uso

## 🚀 Uso

### Interfaz Web
1. Navega a `/image-agent` en la aplicación
2. Ingresa el título y contenido de tu blog
3. Selecciona el estilo y cantidad de imágenes
4. Haz clic en "Generar Imágenes"
5. Visualiza, descarga o regenera las imágenes según necesites

### API REST

#### Generar Imágenes
```bash
POST /api/images/generate
```

```json
{
  "content": "Contenido del blog...",
  "blog_title": "Título del Blog",
  "outline": ["Sección 1", "Sección 2"],
  "style": "professional",
  "count": 3
}
```

#### Regenerar Imagen Específica
```bash
POST /api/images/regenerate/{image_id}
```

```json
{
  "style": "creative",
  "blog_title": "Título del Blog",
  "content": "Contenido del blog..."
}
```

#### Obtener Estilos Disponibles
```bash
GET /api/images/styles
```

## 🏗️ Arquitectura

### Componentes Frontend
- **`ImageAgentPanel.tsx`**: Componente principal para la gestión de imágenes
- **`ImageAgent.tsx`**: Página dedicada al agente de imágenes
- **Integración en Dashboard**: Acceso rápido desde el dashboard principal

### Componentes Backend
- **`ImageAgent`**: Agente especializado en generación de imágenes
- **`images.py`**: Rutas de API para el agente de imágenes
- **Integración en Workflow**: Paso de generación de imágenes en el workflow completo

### Estructura de Datos
```typescript
interface ImageData {
  id: string;
  prompt: string;
  url: string;
  alt_text: string;
  style: string;
  size: string;
}
```

## 🔧 Configuración

### Variables de Entorno
```bash
# Configuración del agente de imágenes
IMAGE_GENERATION_API_URL=https://api.openai.com/v1/images/generations
IMAGE_GENERATION_API_KEY=your_api_key_here
IMAGE_DEFAULT_STYLE=professional
IMAGE_DEFAULT_COUNT=3
```

### Integración con APIs Externas
El agente está preparado para integrarse con:
- **OpenAI DALL-E**
- **Midjourney**
- **Stable Diffusion**
- **Otros servicios de generación de imágenes**

## 📊 Estadísticas y Monitoreo

### Métricas Disponibles
- **Imágenes generadas** por sesión
- **Estilos utilizados** más frecuentemente
- **Tiempo de generación** promedio
- **Tasa de éxito** de generación

### Logs
```python
logger.info(f"ImageAgent executing for blog: {blog_title}")
logger.info(f"Successfully generated {len(images)} images")
logger.error(f"Image generation failed: {error}")
```

## 🧪 Testing

### Script de Demo
```bash
python demo_image_agent.py
```

### Tests Unitarios
```bash
pytest tests/unit/test_image_agent.py
```

### Tests de Integración
```bash
pytest tests/integration/test_image_api.py
```

## 🔄 Workflow Integration

El agente de imágenes se integra perfectamente con el workflow completo:

1. **Planner**: Define el outline del contenido
2. **Researcher**: Investiga el tema
3. **Writer**: Genera el contenido
4. **Editor**: Revisa y mejora el contenido
5. **🖼️ Image Agent**: Genera imágenes para el contenido
6. **SEO**: Optimiza para motores de búsqueda
7. **Social Media**: Adapta para redes sociales

## 🎨 Personalización

### Agregar Nuevos Estilos
```python
# En image_agent.py
def _generate_image_prompts(self, content, blog_title, outline, style):
    # Agregar lógica para nuevos estilos
    if style == "custom":
        # Lógica personalizada
        pass
```

### Modificar Prompts
```python
def _generate_image_prompts(self, content, blog_title, outline, style):
    prompts = []
    
    # Prompt personalizado para imagen principal
    main_prompt = f"Custom prompt for: {blog_title}. Style: {style}"
    prompts.append(main_prompt)
    
    return prompts
```

## 🚀 Próximas Mejoras

### Funcionalidades Planificadas
- [ ] **Integración con APIs reales** de generación de imágenes
- [ ] **Editor de prompts** visual
- [ ] **Filtros avanzados** por estilo y tamaño
- [ ] **Batch processing** para múltiples blogs
- [ ] **Templates predefinidos** para diferentes tipos de contenido
- [ ] **Análisis de calidad** de imágenes generadas

### Optimizaciones Técnicas
- [ ] **Caching** de imágenes generadas
- [ ] **Compresión inteligente** de imágenes
- [ ] **CDN integration** para distribución global
- [ ] **Rate limiting** avanzado
- [ ] **Webhook notifications** para generación completa

## 🤝 Contribución

### Desarrollo Local
1. Clona el repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Configura las variables de entorno
4. Ejecuta el servidor: `uvicorn src.main:app --reload`
5. Ejecuta el frontend: `npm run dev`

### Estructura de Archivos
```
src/
├── agents/specialized/
│   └── image_agent.py          # Agente principal
├── api/routes/
│   └── images.py               # Rutas de API
└── main.py                     # Registro de rutas

frontend/src/
├── components/
│   └── ImageAgentPanel.tsx     # Componente principal
├── pages/
│   └── ImageAgent.tsx          # Página dedicada
└── App.tsx                     # Enrutamiento
```

## 📞 Soporte

Para soporte técnico o preguntas sobre el agente de imágenes:
- 📧 Email: support@credilinq.com
- 📖 Documentación: https://docs.credilinq.com/image-agent
- 🐛 Issues: https://github.com/credilinq/agent/issues

---

**¡Disfruta generando imágenes increíbles para tu contenido! 🎨✨** 