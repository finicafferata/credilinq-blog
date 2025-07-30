#!/bin/bash

echo "🚀 Despliegue Rápido - CrediLinQ Agent"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si estamos en un repositorio Git
if [ ! -d ".git" ]; then
    print_error "No se encontró un repositorio Git. Inicializando..."
    git init
    git add .
    git commit -m "Initial commit - CrediLinQ Agent"
    print_warning "Repositorio Git inicializado. Por favor, conecta con GitHub:"
    echo "git remote add origin https://github.com/tu-usuario/tu-repositorio.git"
    echo "git push -u origin main"
fi

# Verificar si hay cambios sin commitear
if [ -n "$(git status --porcelain)" ]; then
    print_warning "Hay cambios sin commitear. Creando commit..."
    git add .
    git commit -m "Update: Preparación para despliegue $(date)"
fi

# Construir frontend
print_status "Construyendo frontend..."
cd frontend
npm run build
if [ $? -ne 0 ]; then
    print_error "Error al construir el frontend"
    exit 1
fi
cd ..

# Verificar archivos de configuración
print_status "Verificando archivos de configuración..."

if [ ! -f ".env" ]; then
    print_warning "No se encontró archivo .env"
    if [ -f "env.example" ]; then
        cp env.example .env
        print_warning "Archivo .env creado desde env.example"
        print_warning "⚠️  IMPORTANTE: Edita el archivo .env con tus credenciales reales"
    fi
fi

# Crear .gitignore si no existe
if [ ! -f ".gitignore" ]; then
    print_status "Creando .gitignore..."
    cat > .gitignore << EOF
# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Environment variables
.env
.env.local
.env.production

# Build outputs
frontend/dist/
frontend/build/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Database
*.db
*.sqlite
*.sqlite3

# AI/ML
faiss_index/
*.pkl
*.pickle

# Temporary files
*.tmp
*.temp
EOF
fi

print_status "✅ Proyecto listo para despliegue!"
echo ""
echo "📋 Próximos pasos:"
echo ""
echo "1. 🗄️  Configura Supabase:"
echo "   - Ve a https://supabase.com"
echo "   - Crea un nuevo proyecto"
echo "   - Copia DATABASE_URL y SUPABASE_KEY"
echo ""
echo "2. 🔑 Obtén API Keys:"
echo "   - OpenAI: https://platform.openai.com/api-keys"
echo "   - Google AI (opcional): https://makersuite.google.com/app/apikey"
echo ""
echo "3. 🚀 Despliega Backend (Railway):"
echo "   - Ve a https://railway.app"
echo "   - Conecta con GitHub"
echo "   - Selecciona este repositorio"
echo "   - Build Command: pip install -r requirements-vercel.txt"
echo "   - Start Command: uvicorn src.main:app --host 0.0.0.0 --port \$PORT"
echo "   - Agrega variables de entorno desde .env"
echo ""
echo "4. 🎨 Despliega Frontend (Vercel):"
echo "   - Ve a https://vercel.com"
echo "   - Conecta con GitHub"
echo "   - Selecciona este repositorio"
echo "   - Root Directory: frontend"
echo "   - Build Command: npm run build"
echo "   - Output Directory: dist"
echo "   - Agrega VITE_API_URL con la URL de tu backend"
echo ""
echo "5. 🔗 Configura CORS:"
echo "   - Actualiza ALLOWED_ORIGINS en Railway con tu URL de Vercel"
echo "   - Redespliega el backend"
echo ""
echo "📚 Guía completa: DEPLOYMENT_GUIDE.md"
echo "🆘 Si tienes problemas, revisa los logs en Railway/Vercel" 