import tensorflow as tf
import cv2
import numpy as np

def carregar_modelo():
    return tf.keras.models.load_model('modelos/cnn_modelo_cor.h5')

def prever_cor(imagem_path):
    modelo = carregar_modelo()
    imagem = cv2.imread(imagem_path)
    imagem = cv2.resize(imagem, (224, 224))
    imagem = np.expand_dims(imagem, axis=0)
    imagem = imagem / 255.0

    previsao = modelo.predict(imagem)[0]  # Primeiro array contém probabilidades
    classes = ['vermelho', 'azul', 'verde']
    
    for classe, probabilidade in zip(classes, previsao):
        print(f"{classe}: {probabilidade:.2f}")

    indice = np.argmax(previsao)
    return classes[indice]

