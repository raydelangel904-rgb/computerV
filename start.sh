#!/bin/bash

# start.sh - Script de inicialización y despliegue rápido para MoodMeter
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

VENV_DIR=".venv"

echo "=========================================================="
echo "   INICIANDO SISTEMA DE ASISTENCIA FACIAL INTELIGENTE    "
echo "=========================================================="
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "[!] Entorno virtual '.venv' no detectado. Creándolo e instalando dependencias..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --force-reinstall "opencv-python-headless==4.10.0.84"
python -m pip install -r requirements.txt

python - <<'PY'
import cv2
if not hasattr(cv2, 'CascadeClassifier'):
    raise RuntimeError('OpenCV no tiene CascadeClassifier disponible en este entorno.')
print('cv2 OK:', cv2.__version__)
PY

echo ""
echo "[*] Servidor FastAPI arrancando de forma local en http://127.0.0.1:8000"
echo "[*] Abre tu navegador web en esa dirección para ver la interfaz del sistema."
echo "[!] Nota: La primera vez que el backend realice un análisis, DeepFace"
echo "    descargará automáticamente el modelo convolucional de emociones (~20MB)."
echo "    Esto se realiza una sola vez de forma interna."
echo ""
echo "Presiona Ctrl+C para detener el servidor."
echo "----------------------------------------------------------"

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
