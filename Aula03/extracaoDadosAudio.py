## Bibliotecas da aula
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_BLDC = os.path.join(DIR_BASE,"BLDC") ##DIR_BLDC = os.path.join(DIR_BASE,"BLDC\\BLDC")
TRADUCAO = {
    "healthy":"Saudável",
    "propeller":"Falha de Hélice",
    "bearing":"Falha de Rolamento"
}

CLASSES = list(TRADUCAO.keys())
NOMES = list(TRADUCAO.values())

print(DIR_BASE)
print(DIR_BLDC)

"""
Visualização dos dados brutos de um arquivo de áudio
"""
# Função que carrega o dataset
def carregarDataset():
    sinais, rotulos, taxas = [], [], []

    for idx, classe in enumerate(CLASSES):
        caminhoClasse = os.path.join(DIR_BLDC, classe)
        arquivos = sorted([
            f for f in os.listdir(caminhoClasse)
            if f.endswith(".wav")
        ])

        for arquivo in arquivos:
            sinal, sr = librosa.load(
                os.path.join(caminhoClasse, arquivo)
                )
            
            sinais.append(sinal)
            rotulos.append(idx)
            taxas.append(sr)

    return sinais, np.array(rotulos), taxas[0]

sinais, rotulos, SR = carregarDataset()

for idx, nome in enumerate(NOMES):
    print(f"{nome}:{np.sum(rotulos == idx)} arquivos")

fig, eixos = plt.subplots(3, 1, figsize=(12,7), sharex=True)
fig.suptitle("Sinal original de áudio", fontsize=14)
cores = ["#2ca03c","#d62728", "#1f77b4"]

for j, (nome, cor) in enumerate(zip(NOMES, cores)):
    idClasse = np.where(rotulos == j)[0][0]
    tempo = np.linspace(0, len(sinais[idClasse]),len(sinais[idClasse]))
    eixos[j].plot(tempo, sinais[idClasse], color=cor, linewidth=0.5, alpha=0.8)
    eixos[j].set_ylabel(nome, fontsize=10)
    eixos[j].grid(True, alpha=0.3)

eixos[-1].set_xlabel("Tempo (s)")
plt.show()

# Inserir a extracao usando ZCR/RMS

JANELA =2048
SALTO = 512

def extrairZCR(sinal):
    zcr = librosa.feature.zero_crossing_rate(sinal, frame_length=JANELA, hop_length=SALTO)[0]

    return zcr #np.array([zcr.mean(), zcr.std(), zcr.max(), zcr.min()])

atributosZCR = np.array([extrairZCR(s) for s in sinais])

fig, eixos = plt.subplots(3, 1, figsize=(12,7), sharex=True)
fig.suptitle("ZCR do Sinal de áudio", fontsize=14)

for j, (nome, cor) in enumerate(zip(NOMES, cores)):
    idClasse = np.where(rotulos == j)[0][0]
    zcr = atributosZCR[j]
    eixos[j].plot(zcr, color=cor, linewidth=1)
    eixos[j].set_ylabel(nome, fontsize=10)
    eixos[j].grid(True, alpha=0.3)

eixos[-1].set_xlabel("Janela")
plt.tight_layout()
plt.show()

def extrairRMS(sinal):
    rms = librosa.feature.rms(y=sinal, frame_length=JANELA, hop_length=SALTO)[0]

    return rms #np.array([rms.mean(), rms.std(), rms.max(), rms.min()])

atributosRMS = np.array([extrairRMS(s) for s in sinais])

fig, eixos = plt.subplots(3, 1, figsize=(12,7), sharex=True)
fig.suptitle("RMS do Sinal de áudio", fontsize=14)

for j, (nome, cor) in enumerate(zip(NOMES, cores)):
    idClasse = np.where(rotulos == j)[0][0]
    rms = atributosRMS[j]
    eixos[j].plot(rms, color=cor, linewidth=1)
    eixos[j].set_ylabel(nome, fontsize=10)
    eixos[j].grid(True, alpha=0.3)

eixos[-1].set_xlabel("Janela")
plt.tight_layout()
plt.show()
