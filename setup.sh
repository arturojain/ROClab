#!/bin/bash

# Colores para mejor visualización
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}              ROClab - Setup Script             ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Función para verificar si un comando está disponible
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar el sistema operativo
echo -e "${YELLOW}Detectando sistema operativo...${NC}"
OS="$(uname)"
case $OS in
    'Darwin')
        echo -e "${GREEN}macOS detectado.${NC}"
        ;;
    'Linux')
        echo -e "${GREEN}Linux detectado.${NC}"
        ;;
    'MINGW'*|'MSYS'*|'CYGWIN'*)
        echo -e "${GREEN}Windows detectado.${NC}"
        ;;
    *)
        echo -e "${RED}Sistema operativo no reconocido: $OS${NC}"
        echo -e "${YELLOW}El script intentará continuar, pero pueden ocurrir errores.${NC}"
        ;;
esac
echo ""

# Verificar Python
echo -e "${YELLOW}Verificando instalación de Python...${NC}"
if command_exists python3; then
    python_version=$(python3 --version)
    echo -e "${GREEN}Python instalado: $python_version${NC}"
else
    echo -e "${RED}Python 3 no encontrado. Por favor, instala Python 3 antes de continuar.${NC}"
    echo -e "${YELLOW}Puedes descargarlo desde: https://www.python.org/downloads/${NC}"
    exit 1
fi
echo ""

# Directorio del proyecto
PROJECT_DIR="$(pwd)"
echo -e "${YELLOW}Directorio del proyecto: ${GREEN}$PROJECT_DIR${NC}"

# Crear y activar entorno virtual
echo -e "${YELLOW}Creando entorno virtual...${NC}"

# Verificamos si hay caracteres especiales en la ruta
if [[ "$PROJECT_DIR" == *":"* ]]; then
    echo -e "${YELLOW}Se detectaron caracteres especiales en la ruta. Creando entorno virtual en el directorio home...${NC}"
    VENV_DIR="$HOME/roclab_venv"
else
    VENV_DIR="$PROJECT_DIR/.venv"
fi

# Crear entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creando nuevo entorno virtual en $VENV_DIR${NC}"
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error al crear el entorno virtual.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Entorno virtual ya existe en $VENV_DIR${NC}"
fi

# Función para activar el entorno virtual
activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        # Para Windows
        source "$VENV_DIR/Scripts/activate"
    fi
}

# Activar entorno virtual
echo -e "${YELLOW}Activando entorno virtual...${NC}"
activate_venv
if [ $? -ne 0 ]; then
    echo -e "${RED}Error al activar el entorno virtual.${NC}"
    exit 1
fi
echo -e "${GREEN}Entorno virtual activado correctamente.${NC}"
echo ""

# Instalar dependencias
echo -e "${YELLOW}Instalando dependencias...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Error al instalar las dependencias.${NC}"
    exit 1
fi
echo -e "${GREEN}Dependencias instaladas correctamente.${NC}"
echo ""

# Crear script de ejecución
echo -e "${YELLOW}Creando script de ejecución...${NC}"
RUN_SCRIPT="$PROJECT_DIR/run_roclab.sh"

cat > "$RUN_SCRIPT" << EOL
#!/bin/bash
# Script para ejecutar ROClab

# Activar entorno virtual
source "$VENV_DIR/bin/activate"

# Ir al directorio del proyecto
cd "$PROJECT_DIR"

# Ejecutar la aplicación
streamlit run ROClab.py
EOL

chmod +x "$RUN_SCRIPT"
echo -e "${GREEN}Script de ejecución creado en: $RUN_SCRIPT${NC}"
echo ""

# Mensaje final
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}¡Instalación completada con éxito!${NC}"
echo -e "${YELLOW}Para ejecutar ROClab, usa el siguiente comando:${NC}"
echo -e "${BLUE}$RUN_SCRIPT${NC}"
echo -e "${YELLOW}O simplemente:${NC}"
echo -e "${BLUE}./run_roclab.sh${NC}"
echo -e "${BLUE}=================================================${NC}"

# Preguntar si quiere ejecutar la aplicación ahora
read -p "¿Deseas ejecutar la aplicación ahora? (s/n): " execute_now
if [[ "$execute_now" =~ ^[Ss]$ ]]; then
    echo -e "${GREEN}Ejecutando ROClab...${NC}"
    "$RUN_SCRIPT"
else
    echo -e "${YELLOW}Puedes ejecutar la aplicación más tarde con: ./run_roclab.sh${NC}"
fi 