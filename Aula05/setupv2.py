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
# DIR_BASE      = DIR_BASE / "Aula05"
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
def criarUsuario():
    nomeUsuario = input("Digite o Nome do Usuario: ")
    dirUsuario = DIR_DADOS / nomeUsuario
    dirUsuario.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(0)
    contagem = 0

    print(f"=== Cadastrando {nomeUsuario} ===")
    print("Pressione 'q' para sair.")

    while contagem < N_FOTOS:
        ret, frame = cap.read()

        if not ret:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostos = detectorRosto.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, IMG_SIZE)
            caminhoFoto = dirUsuario / f"foto_{contagem:03d}.jpg"
            _,buf = cv2.imencode(".jpg",rosto)
            buf.tofile(str(caminhoFoto))
            contagem += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{nomeUsuario}:{contagem}/{N_FOTOS}", (x,y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Cadastro de Usuario", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Fotos Adquiridas!")

    return contagem

# ── 2. CARREGAMENTO DAS FOTOS CADASTRADAS ─────────────────────────────────────

# ── 3. TREINAMENTO: PCA + REGRESSÃO LOGÍSTICA ────────────────────────────────

# ── MENU PRINCIPAL ────────────────────────────────────────────────────────────
if __name__=="__main__":
    while True:
        print("1. Cadastrar Usuario")
        print("2. Treinar o Modelo")
        print("3. Sair")

        opcao = int(input("\n Selecione sua opção: "))

        if opcao == 1:
            cont = criarUsuario()
        elif opcao == 2:
            pass
        else:
            break





