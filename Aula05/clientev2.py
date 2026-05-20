""" Código cliente — Reconhecimento facial e controle de acesso
    Carrega modelo treinado e valida em tempo real pela câmera
    Autor: Orlando Rosa Junior """

import cv2
import pickle
import numpy as np
from pathlib import Path

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────
DIR_BASE    = Path(__file__).resolve().parent
MODELO_PATH = DIR_BASE / "modelo_acesso.pkl"
HAAR_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── 1. CARREGAR MODELO ────────────────────────────────────────────────────────
if not MODELO_PATH.exists():
    print("Modelo não encontrado!")
    print("Execute primeiro: python 1_setup.py")
    exit(1)

with open(MODELO_PATH, "rb") as f:
    pacote = pickle.load(f)

scaler       = pacote["scaler"]
pca          = pacote["pca"]
modelo       = pacote["modelo"]
nomes        = pacote["nomes"]
imgSize      = pacote["img_size"]
confianca    = pacote["confianca"]
centroides   = pacote.get("centroides", {})
raiosMaximos = pacote.get("raiosMaximos", {})

detectorRosto = cv2.CascadeClassifier(HAAR_PATH)

print(f"Modelo carregado — {len(nomes)} usuário(s): {', '.join(nomes)}")
print("Pressione 'q' para encerrar.\n")

# ── 2. RECONHECIMENTO EM TEMPO REAL ───────────────────────────────────────────
