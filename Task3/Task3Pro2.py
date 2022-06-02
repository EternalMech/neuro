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
import scipy

# Загружаем датасет автомобилей
url = "https://drive.google.com/uc?id=1IfpCPk8QJjRg-xjKWxM_-8WTwtoEAQky"
output = "Автомобили.zip"
gdown.download(url, output)

# Распаковываем и переименовываем
with zipfile.ZipFile("Автомобили.zip","r") as zip_ref:
    zip_ref.extractall()
os.rename('ÇóΓ«¼«í¿½¿', 'Auto')

# Формируем генератор изображений из тренировочных и проверочных данных
datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

train_generator = datagen.flow_from_directory(
    'Auto/train',
    target_size=(54, 96),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'Auto/val',
    target_size=(54, 96),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)


model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=(54, 96, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 16,
    validation_data = val_generator,
    validation_steps = val_generator.samples // 16,
    epochs=10,
    verbose=1
)

#Оображаем график точности обучения
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

