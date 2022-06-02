from tensorflow.keras.datasets import cifar10  # Загружаем базу cifar10
from tensorflow.keras.models import Sequential  # Сеть прямого распространения
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # работа с изображениями
from tensorflow.keras.optimizers import Adam, Adadelta  # оптимизаторы
from tensorflow.keras import utils  # Используем дял to_categoricall
from tensorflow.keras.preprocessing import image  # Для отрисовки изображений
import numpy as np  # Библиотека работы с массивами
import matplotlib.pyplot as plt  # Для отрисовки графиков
from PIL import Image  # Для отрисовки изображений
import random  # Для генерации случайных чисел
import math  # Для округления
import os  # Для работы с файлами
import gdown
import zipfile

# Загружаем данные cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)

# Трансформируем данные в ohe
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

print(x_train.shape, y_train.shape)

# Создаем модель нейронной сети, подбирая эффективные слои и параметры
modelCifar10 = Sequential()

modelCifar10.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.3))

modelCifar10.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
modelCifar10.add(MaxPooling2D(pool_size=(2, 2)))

modelCifar10.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.4))

modelCifar10.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
modelCifar10.add(MaxPooling2D(pool_size=(2, 2)))

modelCifar10.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.4))

modelCifar10.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.4))

modelCifar10.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
modelCifar10.add(MaxPooling2D(pool_size=(2, 2)))

modelCifar10.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.4))

modelCifar10.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
modelCifar10.add(Dropout(0.4))

modelCifar10.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
modelCifar10.add(MaxPooling2D(pool_size=(2, 2)))

modelCifar10.add(Flatten())
modelCifar10.add(Dropout(0.5))
modelCifar10.add(Dense(512, activation='relu'))
modelCifar10.add(BatchNormalization())
modelCifar10.add(Dropout(0.5))
modelCifar10.add(Dense(10, activation='softmax'))

modelCifar10.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучаем нейронную сеть
history = modelCifar10.fit(x_train, y_train, batch_size=64, epochs=40, verbose=1, validation_data=(x_test, y_test))

# Выводим график ошибок
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

# Выводим график точности
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

# Проверим работу нейронной сети на тестовой выборке
classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

rand = random.randint(0, 9999)
plt.imshow(Image.fromarray(x_test[rand]).convert('RGBA'))
plt.show()
print('Реальный образ: ', classes[np.argmax(y_test[rand])])
print('Распознанный образ: ', classes[np.argmax(modelCifar10.predict(x_test[rand].reshape(1, 32, 32, 3)))])
print()

rand = random.randint(0, 9999)
plt.imshow(Image.fromarray(x_test[rand]).convert('RGBA'))
plt.show()
print('Реальный образ: ', classes[np.argmax(y_test[rand])])
print('Распознанный образ: ', classes[np.argmax(modelCifar10.predict(x_test[rand].reshape(1, 32, 32, 3)))])
print()

rand = random.randint(0, 9999)
plt.imshow(Image.fromarray(x_test[rand]).convert('RGBA'))
plt.show()
print('Реальный образ: ', classes[np.argmax(y_test[rand])])
print('Распознанный образ: ', classes[np.argmax(modelCifar10.predict(x_test[rand].reshape(1, 32, 32, 3)))])
print()

rand = random.randint(0, 9999)
plt.imshow(Image.fromarray(x_test[rand]).convert('RGBA'))
plt.show()
print('Реальный образ: ', classes[np.argmax(y_test[rand])])
print('Распознанный образ: ', classes[np.argmax(modelCifar10.predict(x_test[rand].reshape(1, 32, 32, 3)))])
print()