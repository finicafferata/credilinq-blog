# 🚀 Guía de Despliegue - CrediLinQ Agent

Esta guía te ayudará a publicar tu aplicación CrediLinQ Agent desde cero.

## 📋 Prerrequisitos

- Cuenta en [GitHub](https://github.com)
- Cuenta en [Vercel](https://vercel.com) (gratis)
- Cuenta en [Railway](https://railway.app) o [Render](https://render.com) (gratis)
- Cuenta en [Supabase](https://supabase.com) (gratis)
- API Key de [OpenAI](https://openai.com)

## 🗄️ Paso 1: Configurar Base de Datos (Supabase)

1. Ve a [Supabase](https://supabase.com) y crea una cuenta
2. Crea un nuevo proyecto
3. Ve a **Settings > Database** y copia la URL de conexión
4. Ve a **Settings > API** y copia la URL y anon key

## 🔧 Paso 2: Preparar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con:

```bash
# Database Configuration
DATABASE_URL="postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"
SUPABASE_URL="https://[project-ref].supabase.co"
SUPABASE_KEY="[your-anon-key]"

# OpenAI Configuration
OPENAI_API_KEY="sk-..."

# Google AI Configuration (opcional)
GOOGLE_API_KEY="..."

# Application Settings
ENVIRONMENT="production"
API_VERSION="2.0.0"
SECRET_KEY="tu-clave-secreta-aqui"

# CORS Settings (actualizar después del despliegue)
ALLOWED_ORIGINS="https://tu-app.vercel.app,http://localhost:3000"
```

## 🚀 Paso 3: Desplegar Backend en Railway

### Opción A: Railway (Recomendado)

1. Ve a [Railway](https://railway.app) y conéctate con GitHub
2. Crea un nuevo proyecto
3. Selecciona "Deploy from GitHub repo"
4. Selecciona tu repositorio
5. En la configuración del servicio:
   - **Build Command**: `pip install -r requirements-vercel.txt`
   - **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
6. Ve a **Variables** y agrega todas las variables de `.env`
7. Railway detectará automáticamente que es una app Python

### Opción B: Render

1. Ve a [Render](https://render.com) y conéctate con GitHub
2. Crea un nuevo **Web Service**
3. Selecciona tu repositorio
4. Configura:
   - **Build Command**: `pip install -r requirements-vercel.txt`
   - **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Agrega las variables de entorno en **Environment**

## 🎨 Paso 4: Desplegar Frontend en Vercel

1. Ve a [Vercel](https://vercel.com) y conéctate con GitHub
2. Crea un nuevo proyecto
3. Selecciona tu repositorio
4. En la configuración:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. En **Environment Variables**, agrega:
   ```
   VITE_API_URL=https://tu-backend.railway.app
   ```
6. Despliega

## 🔗 Paso 5: Configurar CORS

Una vez que tengas las URLs de tu frontend y backend:

1. Actualiza `ALLOWED_ORIGINS` en tu backend con la URL de Vercel
2. Actualiza `VITE_API_URL` en Vercel con la URL de tu backend
3. Redespliega ambos servicios

## 🧪 Paso 6: Verificar Despliegue

1. **Backend**: Visita `https://tu-backend.railway.app/docs`
2. **Frontend**: Visita tu URL de Vercel
3. **API Health**: Visita `https://tu-backend.railway.app/health`

## 🔧 Solución de Problemas

### Error: "Module not found"
- Verifica que todas las dependencias estén en `requirements-vercel.txt`
- Asegúrate de que el build command instale las dependencias

### Error: "Database connection failed"
- Verifica que `DATABASE_URL` esté correcta
- Asegúrate de que Supabase esté activo

### Error: "CORS error"
- Actualiza `ALLOWED_ORIGINS` con la URL correcta de tu frontend
- Redespliega el backend

### Error: "API key not found"
- Verifica que `OPENAI_API_KEY` esté configurada
- Asegúrate de que la API key sea válida

## 📊 Monitoreo

- **Railway**: Dashboard con logs y métricas
- **Vercel**: Analytics y performance
- **Supabase**: Database logs y queries

## 🔄 Actualizaciones

Para actualizar tu aplicación:

1. Haz push a tu repositorio de GitHub
2. Railway y Vercel se actualizarán automáticamente
3. Verifica que todo funcione correctamente

## 💰 Costos Estimados

- **Vercel**: Gratis (hasta 100GB bandwidth)
- **Railway**: Gratis (hasta $5/mes)
- **Supabase**: Gratis (hasta 500MB database)
- **OpenAI**: ~$0.01-0.10 por request

## 🆘 Soporte

Si tienes problemas:
1. Revisa los logs en Railway/Vercel
2. Verifica las variables de entorno
3. Prueba localmente primero
4. Consulta la documentación de cada plataforma 