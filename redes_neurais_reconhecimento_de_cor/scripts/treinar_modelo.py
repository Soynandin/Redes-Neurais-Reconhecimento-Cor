import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocessamento import preprocessar
import os

def criar_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def treinar():

    if not os.path.exists('modelos'):
        os.makedirs('modelos')
        print("Diretório 'modelos' criado com sucesso.")

    X_train, X_val, y_train, y_val = preprocessar()
    print(f"Dados carregados. X_train: {X_train.shape}, y_train: {y_train.shape}")

    modelo = criar_modelo()
    print("Modelo criado. Iniciando o treinamento...")

    modelo.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    print("Treinamento finalizado. Salvando o modelo...")

    try:
        modelo.save('modelos/cnn_modelo_cor.h5', save_format='h5')
        if os.path.exists('modelos/cnn_modelo_cor.h5'):
            print("Modelo salvo com sucesso!")
        else:
            print("Erro: O modelo não foi salvo.")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")

if __name__ == "__main__":
    treinar()
