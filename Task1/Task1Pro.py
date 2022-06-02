from tensorflow.keras.datasets import mnist     #Библиотека с базой Mnist
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout       # Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam    # Подключаем оптимизатор Adam
from tensorflow.keras import utils              #Утилиты для to_categorical
import numpy as np                              # Подключаем библиотеку numpy
from tensorflow.keras.preprocessing import image #Для отрисовки изображения
import matplotlib.pyplot as plt # Отрисовка графика
from PIL import Image
import os

import tensorflow as tf
import random


seed = 256
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Здесь я обучу нейронку, а в следующем разделе проверю ее

# Загрузка данных Mnist
(x_train_org, y_train_org), (x_test, y_test) = mnist.load_data()

# Преобразуем тренировочные данные для нейронной сети
x_train = x_train_org.reshape(60000, 784).astype('float32') / 255
y_train = utils.to_categorical(y_train_org, 10)

#{'epoches': 35, 'activation': 'relu', 'batch_size': 29585, '1LayerNeurons': 1344, '2LayerNeurons': 1680}. Best is trial 16 with value: 0.9749000072479248.

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(1344, input_dim=784, activation='relu'))
model.add(Dense(1680, activation='relu'))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучаем нейронную сеть
history = model.fit(x_train,
                    y_train,
                    batch_size=29585,
                    epochs=35,
                    verbose=1,
                    validation_split = 0.3)

# Выводим график ошибок
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

# Выводим график точности
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


# Здесь я проверю эффективность нейронной сети на тестовых данных

# Преобразуем тестовые данные для нейронной сети
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_test = utils.to_categorical(y_test, 10)

# Оценим модель на тестовых данных
results = model.evaluate(x_test, y_test, batch_size=128)
print('Тестовая ошибка:', round(results[0], 4))
print('Точность на тестовой выборке: ', round(results[1]*100, 2), '%', sep='')


# Здесь я протестирую обученную нейронную сеть на своих изображениях

# Загружаем картинки(Paint и Photo) с изображенной цифрой и конвертируем их в массив
imgPaint = image.load_img('content/TestNumber.png', target_size=(28, 28), color_mode='grayscale')
imgPhoto = image.load_img('content/Photo.jpg', target_size=(28, 28), color_mode='grayscale')
imgArrPaint = image.img_to_array(imgPaint)
imgArrPhoto = image.img_to_array(imgPhoto)

# Преобразуем массив для нейронной сети
imgArrPaint = imgArrPaint.reshape(1, 784)
imgArrPhoto = imgArrPhoto.reshape(1, 784)
imgArrPaint[0] = [abs(i - 255) / 255 for i in imgArrPaint[0]]
imgArrPhoto[0] = [abs(i - 255) / 255 for i in imgArrPhoto[0]]

# Используем нейронную сеть, чтобы узнать что было на картинках
resultPaint = model.predict(imgArrPaint)
resultPhoto = model.predict(imgArrPhoto)
print('На изображениях были следующие числа: ')

plt.imshow(imgPaint, cmap='gray')
plt.show()
print(np.argmax(resultPaint))
plt.imshow(imgPhoto, cmap='gray')
plt.show()
print(np.argmax(resultPhoto))

results = model.evaluate(x_test, y_test, batch_size=128)

print('Точность по Optuna: 0.9749000072479248')
print('Точность нейронной сети: ', results[1])

# Оно работает! :)
# +моральное удовлетворение от проделанной работы