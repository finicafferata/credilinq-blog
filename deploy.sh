#!/bin/bash

echo "🚀 Iniciando despliegue de CrediLinQ Agent..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "package.json" ]; then
    print_error "No se encontró package.json. Asegúrate de estar en el directorio raíz del proyecto."
    exit 1
fi

print_status "Verificando dependencias..."

# Verificar si Node.js está instalado
if ! command -v node &> /dev/null; then
    print_error "Node.js no está instalado. Por favor instálalo primero."
    exit 1
fi

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado. Por favor instálalo primero."
    exit 1
fi

print_status "Instalando dependencias del frontend..."
cd frontend
npm install
if [ $? -ne 0 ]; then
    print_error "Error al instalar dependencias del frontend"
    exit 1
fi

print_status "Construyendo frontend..."
npm run build
if [ $? -ne 0 ]; then
    print_error "Error al construir el frontend"
    exit 1
fi

cd ..

print_status "Verificando variables de entorno..."
if [ ! -f ".env" ]; then
    print_warning "No se encontró archivo .env. Crea uno basado en env.example"
    print_status "Copiando env.example a .env..."
    cp env.example .env
    print_warning "Por favor, edita el archivo .env con tus credenciales reales"
fi

print_status "Despliegue preparado correctamente!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Configura las variables de entorno en tu plataforma de despliegue"
echo "2. Despliega el backend en Railway o Render"
echo "3. Despliega el frontend en Vercel"
echo ""
echo "🔗 Enlaces útiles:"
echo "- Railway: https://railway.app"
echo "- Vercel: https://vercel.com"
echo "- Render: https://render.com" 