import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset MNIST
# x_treino = imagens de treino, y_treino = rótulos de treino
(x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

# Normalizar os dados
# As imagens são compostas por valores de pixel entre 0 e 255
# Dividimos por 255 para que os valores fiquem entre 0 e 1
x_treino = x_treino / 255.0
x_teste = x_teste / 255.0

# Verificar as dimensões dos dados
# x_treino deve ter a forma (60000, 28, 28)
# y_treino deve ter a forma (60000,)
print(x_treino.shape)
print(y_treino.shape)

# Construir o modelo
# Usamos uma rede neural simples com uma camada densa oculta
# A camada de entrada é achatada para transformar a imagem 28x28 em um vetor de 784 elementos
# 'relu' é a função de ativação que introduz não-linearidade, permite que o modelo aprenda relações complexas nos dados
# A camada de saída tem 10 neurônios, um para cada dígito (0-9)
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar e treinar o modelo
# Usamos 'sparse_categorical_crossentropy' porque os rótulos são inteiros
# 'adam' é um otimizador eficiente para este tipo de problema
# 'accuracy' é a métrica que queremos monitorar durante o treinamento
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
# Usamos 10 épocas para garantir que o modelo tenha tempo suficiente para aprender
modelo.fit(x_treino, y_treino, epochs=10)

# Avaliar o modelo
# Avaliamos o modelo no conjunto de teste e imprimimos a precisão
test_loss, test_acu = modelo.evaluate(x_teste, y_teste, verbose=2)
print(f'\nPrecisão no conjunto de teste: {test_acu}')

# Fazer previsões
# Usamos o modelo treinado para fazer previsões no conjunto de teste
predicao = modelo.predict(x_teste)

# Prever a primeira imagem de teste
print(np.argmax(predicao[0]))

# Visualizar a imagem
# Mostramos a primeira imagem do conjunto de teste para verificar a previsão
plt.imshow(x_teste[0], cmap=plt.cm.Purples)
plt.show()

# Prever a segunda imagem de teste
# Mostramos a segunda imagem do conjunto de teste para verificar a previsão
print(np.argmax(predicao[1]))
plt.imshow(x_teste[1], cmap=plt.cm.Purples)
plt.show()
