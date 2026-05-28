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
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cinza  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = detectorRosto.detectMultiScale(
        cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in rostos:
        # Pré-processamento — mesmo pipeline do setup
        rosto    = cinza[y:y+h, x:x+w]
        rosto    = cv2.resize(rosto, imgSize)
        vetor    = rosto.flatten().astype(float).reshape(1, -1)
        vetorNorm = scaler.transform(vetor)
        vetorPCA  = pca.transform(vetorNorm)

        # Classificação
        probabilidades = modelo.predict_proba(vetorPCA)[0]
        idxMelhor      = np.argmax(probabilidades)
        confMelhor     = probabilidades[idxMelhor]
        nomeDetectado  = nomes[idxMelhor]

        # Verificação por distância ao centroide
        distCentroide = np.linalg.norm(
            vetorPCA[0] - centroides.get(idxMelhor, vetorPCA[0])
        )
        raioMax = raiosMaximos.get(idxMelhor, float("inf"))
        dentroDaDistancia = distCentroide <= raioMax

        # Resultado — exige AMBOS: confiança alta E distância aceitável
        if confMelhor >= confianca and dentroDaDistancia:
            label = f"Acesso liberado: {nomeDetectado}"
            cor   = (0, 200, 0)
        else:
            label = "Acesso negado"
            cor   = (0, 0, 220)

        # Exibir resultado na tela
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), cor, -1)
        cv2.putText(frame, label,
                    (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Confianca: {confMelhor*100:.0f}%",
                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, cor, 1)

    cv2.putText(frame, "Sistema de Controle de Acesso",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.imshow("Controle de Acesso", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistema encerrado.")