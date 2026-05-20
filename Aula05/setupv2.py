""" Código de setup — Cadastro de usuários e treinamento do modelo
    Pipeline: câmera → detecção de rosto → PCA → Regressão Logística
    Autor: Orlando Rosa Junior """

import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────
DIR_BASE      = Path(__file__).resolve().parent
DIR_DADOS     = DIR_BASE / "dados_usuarios"
MODELO_PATH   = DIR_BASE / "modelo_acesso.pkl"
IMG_SIZE      = (64, 64)   # tamanho de cada rosto capturado
N_FOTOS       = 100        # fotos capturadas por usuário
N_COMPONENTES = 50         # componentes PCA
CONFIANCA_MIN = 0.62       # limiar mínimo de confiança para aceitar

# Detector de rosto do OpenCV (Haar Cascade)
HAAR_PATH     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detectorRosto = cv2.CascadeClassifier(HAAR_PATH)

DIR_DADOS.mkdir(exist_ok=True)

# ── 1. CADASTRO DE USUÁRIOS ───────────────────────────────────────────────────


# ── 2. CARREGAMENTO DAS FOTOS CADASTRADAS ─────────────────────────────────────

# ── 3. TREINAMENTO: PCA + REGRESSÃO LOGÍSTICA ────────────────────────────────

# ── MENU PRINCIPAL ────────────────────────────────────────────────────────────
