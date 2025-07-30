# 🗄️ Guía de Configuración de Supabase

## 📋 Estado Actual

Según la verificación, tu configuración actual es:

- ✅ **DATABASE_URL**: Configurada (pero con valores de ejemplo)
- ❌ **SUPABASE_URL**: Necesita configuración
- ❌ **SUPABASE_KEY**: Necesita configuración  
- ❌ **OPENAI_API_KEY**: Necesita configuración

## 🚀 Pasos para Configurar Supabase

### 1. Crear Proyecto en Supabase

1. Ve a [https://supabase.com](https://supabase.com)
2. Crea una cuenta o inicia sesión
3. Haz clic en **"New Project"**
4. Completa la información:
   - **Name**: `credilinq-agent`
   - **Database Password**: Crea una contraseña segura
   - **Region**: Selecciona la más cercana a ti
5. Haz clic en **"Create new project"**

### 2. Obtener Credenciales

1. Una vez creado el proyecto, ve a **Settings > API**
2. Copia los siguientes valores:
   - **Project URL**: `https://[project-id].supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### 3. Configurar Variables de Entorno

Edita tu archivo `.env` con los valores reales:

```bash
# Database Configuration
DATABASE_URL="postgresql://postgres:[TU_CONTRASEÑA]@db.[project-id].supabase.co:5432/postgres"
SUPABASE_URL="https://[project-id].supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# OpenAI Configuration
OPENAI_API_KEY="sk-..." # Obtén de https://platform.openai.com/api-keys
```

### 4. Verificar Configuración

Ejecuta el script de verificación:

```bash
python3 check_supabase_status.py
```

### 5. Probar Conexión

Una vez configurado, prueba la conexión:

```bash
python3 test_supabase_connection.py
```

## 🔧 Scripts Disponibles

### Verificar Estado Actual
```bash
python3 check_supabase_status.py
```

### Configuración Interactiva
```bash
python3 setup_supabase.py
```

### Probar Conexión
```bash
python3 test_supabase_connection.py
```

## 📊 Estructura de Base de Datos

Una vez configurado, Supabase creará automáticamente las siguientes tablas:

- `blogs` - Posts del blog
- `campaigns` - Campañas de marketing
- `campaign_tasks` - Tareas de campaña
- `documents` - Documentos procesados
- `analytics` - Métricas y analytics

## 🔐 Seguridad

- ✅ **SSL/TLS**: Conexiones encriptadas
- ✅ **Row Level Security**: Control de acceso por fila
- ✅ **API Keys**: Autenticación segura
- ✅ **Backup automático**: Cada 24 horas

## 💰 Costos

- **Plan Gratuito**: 
  - 500MB de base de datos
  - 2GB de transferencia
  - 50,000 requests/mes
  - Perfecto para desarrollo y proyectos pequeños

## 🆘 Solución de Problemas

### Error: "Connection refused"
- Verifica que la URL de Supabase sea correcta
- Asegúrate de que el proyecto esté activo

### Error: "Authentication failed"
- Verifica que la anon key sea correcta
- Asegúrate de copiar la key completa

### Error: "Database does not exist"
- El proyecto se crea automáticamente
- Espera unos minutos después de crear el proyecto

## 📞 Soporte

- **Documentación**: [docs.supabase.com](https://docs.supabase.com)
- **Discord**: [discord.supabase.com](https://discord.supabase.com)
- **GitHub**: [github.com/supabase/supabase](https://github.com/supabase/supabase)

---

**¡Una vez configurado Supabase, tu aplicación estará lista para el despliegue! 🎉** 