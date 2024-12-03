import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def carregar_imagens(diretorio):
    imagens = []
    labels = []
    label_encoder = LabelEncoder()

    for label in os.listdir(diretorio):
        label_path = os.path.join(diretorio, label)
        if os.path.isdir(label_path):
            for imagem_nome in os.listdir(label_path):
                imagem_path = os.path.join(label_path, imagem_nome)
                if imagem_nome.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"Carregando imagem: {imagem_path}")
                    imagem = carregar_e_preprocessar_imagem(imagem_path)
                    imagens.append(imagem)
                    labels.append(label)
    labels = label_encoder.fit_transform(labels)
    return np.array(imagens), np.array(labels)

def carregar_e_preprocessar_imagem(caminho_imagem):
    imagem = Image.open(caminho_imagem).convert('RGB')
    imagem = imagem.resize((224, 224))  # 224x224
    imagem = np.array(imagem) / 255.0
    return imagem

def preprocessar():
    diretorio = "dados\\treino"

    if not os.path.exists(diretorio):
        raise FileNotFoundError(f"O diretório {diretorio} não existe.")

    imagens, labels = carregar_imagens(diretorio)

    if len(imagens) == 0 or len(labels) == 0:
        raise ValueError("Nenhuma imagem ou rótulo foi carregado. Verifique os dados de entrada.")

    print(f"Imagens carregadas: {len(imagens)}")
    print(f"Labels carregados: {len(labels)}")
    
    return train_test_split(imagens, labels, test_size=0.2, random_state=42)
