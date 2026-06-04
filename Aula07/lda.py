import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

# Carregamento dos dados
DIR = Path(__file__).resolve().parent #C:\Temp\Edwillie-senai_iat2_rp\Aula07

def rglob(nomeArquivo):
    return list(DIR.rglob(nomeArquivo))[0]

xTreino = np.loadtxt(str(rglob("X_train.txt")))
yTreino = np.loadtxt(str(rglob("y_train.txt")), dtype=int)

xTeste = np.loadtxt(str(rglob("X_test.txt")))
yTeste = np.loadtxt(str(rglob("y_test.txt")), dtype=int)

NOMES = ["Caminhando", "Subindo escadas", "Descendo Escadas", "Sentado", "Em Pé", "Deitado"]
Cores = ["green", "blue", "red", "black", "yellow", "purple"]

print(f"Treino: {xTreino.shape} Teste: {xTeste.shape}")

# Normalizacao
xescala = RobustScaler()
xTreinoNormalizado = xescala.fit_transform(xTreino)
xTesteNormalizado  = xescala.transform(xTeste)

#PCA
pca = PCA(n_components=5)
xTreinoPCA = pca.fit_transform(xTreinoNormalizado)
xTestePCA = pca.fit_transform(xTesteNormalizado)

print(f"PCA para 5 elementos: {pca.explained_variance_ratio_.sum()*100:.1f} % da variancia")

#LDA
lda = LinearDiscriminantAnalysis()
xTreinoLDA = lda.fit_transform(xTreinoNormalizado, yTreino)
xTesteLDA = lda.transform(xTesteNormalizado)

print(f"LDA para 5 elementos: {lda.explained_variance_ratio_.sum()*100:.1f} % da variancia")

# Analise Comparativa
varAcumPCA = np.cumsum(pca.explained_variance_ratio_)*100
varAcumLDA = np.cumsum(lda.explained_variance_ratio_)*100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.plot(range(1, len(varAcumPCA) + 1), varAcumPCA, marker='o', color="red", linewidth=2)
ax1.set_title("PCA")
ax1.set_xlabel("Numero do componente")
ax1.set_ylabel("Variancia Acumulada (%)")
ax1.set_ylim(0,110)
ax1.axhline(80, color="gray", linestyle="--", linewidth=1)
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(varAcumLDA) + 1) , np.diff(np.concatenate([[0],varAcumLDA])))
ax2.plot(range(1, len(varAcumLDA) + 1) , varAcumLDA, marker='o', color="red", linewidth=2)
ax2.set_title("LDA")
ax2.set_xlabel("Numero do componente")
ax2.set_ylabel("Variancia Acumulada (%)")
ax2.set_ylim(0,110)
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()