#!/bin/bash
# Script para ejecutar ROClab

# Activar entorno virtual
source "/Users/gustavo/roclab_venv/bin/activate"

# Ir al directorio del proyecto
cd "/Users/gustavo/Dev/Research/OMINIS : Didactiva ROCLab/roclab.ominis.org"

# Ejecutar la aplicación
streamlit run ROClab.py
