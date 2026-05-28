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
DIR_BASE     = Path(__file__).resolve().parent
DIR_DADOS    = DIR_BASE / "dados_usuarios"
MODELO_PATH  = DIR_BASE / "modelo_acesso.pkl"
IMG_SIZE     = (64, 64)      # tamanho de cada rosto capturado
N_FOTOS      = 100            # fotos capturadas por usuário
N_COMPONENTES = 50           # componentes PCA
CONFIANCA_MIN = 0.62       # limiar mínimo de confiança para aceitar

# Detector de rosto do OpenCV (Haar Cascade)
HAAR_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detectorRosto = cv2.CascadeClassifier(HAAR_PATH)

DIR_DADOS.mkdir(exist_ok=True)

# ── 1. CADASTRO DE USUÁRIOS ───────────────────────────────────────────────────
def cadastrarUsuario(nome):
    """Abre a câmera e captura N_FOTOS do rosto do usuário."""
    dirUsuario = DIR_DADOS / nome
    dirUsuario.mkdir(exist_ok=True)

    cap      = cv2.VideoCapture(0)
    contagem = 0

    print(f"\nCadastrando: {nome}")
    print(f"Posicione o rosto na câmera. Serão capturadas {N_FOTOS} fotos.")
    print("Pressione 'q' para cancelar.\n")

    while contagem < N_FOTOS:
        ret, frame = cap.read()
        if not ret:
            break

        cinza  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detectorRosto.detectMultiScale(
            cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in rostos:
            rosto = cinza[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, IMG_SIZE)

            # Usar imencode + tofile — evita falha com acentos no Windows
            caminhoFoto = dirUsuario / f"foto_{contagem:03d}.jpg"
            _, buf = cv2.imencode(".jpg", rosto)
            buf.tofile(str(caminhoFoto))
            contagem += 1

            # Desenhar retângulo e progresso
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{contagem}/{N_FOTOS}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Cadastrando: {nome}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("Cadastro", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  {contagem} fotos salvas para {nome}.")
    return contagem

# ── 2. CARREGAMENTO DAS FOTOS CADASTRADAS ─────────────────────────────────────
def carregarDados():
    """Lê todas as fotos salvas e retorna vetores e rótulos."""
    xDados, yDados, nomes = [], [], []
    usuarios = sorted([d for d in DIR_DADOS.iterdir() if d.is_dir()])

    if not usuarios:
        print("Nenhum usuário cadastrado.")
        return None, None, None

    for idx, dirUsuario in enumerate(usuarios):
        fotos = list(dirUsuario.glob("*.jpg"))
        nomes.append(dirUsuario.name)
        for foto in fotos:
            # Usar numpy para ler bytes — evita falha com acentos no Windows
            buf = np.fromfile(str(foto), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                xDados.append(img.flatten().astype(float))
                yDados.append(idx)

    print(f"\nDados carregados:")
    for i, nome in enumerate(nomes):
        n = sum(1 for y in yDados if y == i)
        print(f"  {nome}: {n} fotos")

    return np.array(xDados), np.array(yDados), nomes

# ── 3. TREINAMENTO: PCA + REGRESSÃO LOGÍSTICA ────────────────────────────────
def treinarModelo(xDados, yDados, nomes):
    """Aplica PCA e treina Regressão Logística."""

    # Normalização
    scaler = StandardScaler()
    xNorm  = scaler.fit_transform(xDados)

    # PCA — reduz de 4.096 para N_COMPONENTES dimensões
    nComp  = min(N_COMPONENTES, xDados.shape[0] - 1, xDados.shape[1])
    pca    = PCA(n_components=nComp, whiten=True, random_state=42)
    xPCA   = pca.fit_transform(xNorm)

    varTotal = pca.explained_variance_ratio_.sum()
    print(f"\nPCA: {nComp} componentes → {varTotal*100:.1f}% da variância explicada")

    # Divisão treino/teste
    xTrain, xTest, yTrain, yTest = train_test_split(
        xPCA, yDados, test_size=0.2, random_state=42, stratify=yDados
    )

    # Regressão Logística
    modelo = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    modelo.fit(xTrain, yTrain)

    # Avaliação
    acc = accuracy_score(yTest, modelo.predict(xTest))
    print(f"Acurácia no teste: {acc*100:.1f}%")
    print("\nRelatório por usuário:")
    print(classification_report(yTest, modelo.predict(xTest),
                                  target_names=nomes))

    # Calcular centroide e raio máximo de cada usuário no espaço PCA
    # Usado no cliente para rejeitar rostos desconhecidos por distância
    centroides   = {}
    raiosMaximos = {}
    for idx, nome in enumerate(nomes):
        pontosUsuario     = xPCA[yDados == idx]
        centroide         = pontosUsuario.mean(axis=0)
        distancias        = np.linalg.norm(pontosUsuario - centroide, axis=1)
        centroides[idx]   = centroide
        # Raio = média + 2 desvios padrão — cobre ~95% dos rostos cadastrados
        raiosMaximos[idx] = distancias.mean() + 2 * distancias.std()
        print(f"  {nome}: raio máximo = {raiosMaximos[idx]:.2f}")

    # Salvar modelo
    pacote = {
        "scaler":       scaler,
        "pca":          pca,
        "modelo":       modelo,
        "nomes":        nomes,
        "img_size":     IMG_SIZE,
        "confianca":    CONFIANCA_MIN,
        "centroides":   centroides,
        "raiosMaximos": raiosMaximos,
    }
    with open(MODELO_PATH, "wb") as f:
        pickle.dump(pacote, f)

    print(f"Modelo salvo em: {MODELO_PATH}")
    return acc

# ── MENU PRINCIPAL ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Sistema de Controle de Acesso — Setup")
    print("=" * 50)

    while True:
        print("\n1. Cadastrar novo usuário")
        print("2. Treinar modelo com usuários cadastrados")
        print("3. Sair")
        opcao = input("\nEscolha: ").strip()

        if opcao == "1":
            nome = input("Nome do usuário: ").strip()
            if nome:
                cadastrarUsuario(nome)
            else:
                print("Nome inválido.")

        elif opcao == "2":
            xDados, yDados, nomes = carregarDados()
            if xDados is not None and len(nomes) >= 1:
                treinarModelo(xDados, yDados, nomes)
            elif xDados is not None and len(nomes) < 1:
                print("Cadastre pelo menos 2 usuários antes de treinar.")

        elif opcao == "3":
            print("Encerrando setup.")
            break