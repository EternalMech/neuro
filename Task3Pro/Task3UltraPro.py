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

# Для этой задачки я скачал изображения кошек и собак и буду обучать
# нейронную сеть отличать кошку от собаки

# Загружаем датасет автомобилей
url = "https://drive.google.com/uc?id=12d4l1sgfXStBKsIEza2OjGvau06xN8ZK"
output = "CatDogs.zip"
gdown.download(url, output)

with zipfile.ZipFile("CatDogs.zip","r") as zip_ref:
    zip_ref.extractall()


# Задаем параметры генератору изображений
datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    validation_split=0.1
)

# Формируем тренировочные данные
train_generator = datagen.flow_from_directory(
    'PetImages',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Формируем проверочные данные
val_generator = datagen.flow_from_directory(
    'PetImages',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# Создаем модель нейронной сети
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(300, 300, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(BatchNormalization())

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(BatchNormalization())

model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32,
    epochs=15,
    verbose=1)

# Скачиваем тестовые данные, чтобы проверить нейронную сеть
url = "https://drive.google.com/uc?id=19ejAQpnMnJEN9AyiCT8FbTkKzkoTW4Xc"
output = "testdata.zip"
gdown.download(url, output)

with zipfile.ZipFile("testdata.zip", "r") as zip_ref:
    zip_ref.extractall()

# Обрабатываем изображение кота и собаки
imgCat = image.load_img('testdata/cat.jpg', target_size=(300, 300))
imgArrCat = image.img_to_array(imgCat)
imgArrCat *= 1. / 255
imgArrCat = imgArrCat[None]

imgDoge = image.load_img('testdata/doge.jpg', target_size=(300, 300))
imgArrDoge = image.img_to_array(imgDoge)
imgArrDoge *= 1. / 255
imgArrDoge = imgArrDoge[None]

# Ради интереса попробуем обработать тигра и волка :)
imgTiger = image.load_img('testdata/tiger.jpg', target_size=(300, 300))
imgArrTiger = image.img_to_array(imgTiger)
imgArrTiger *= 1. / 255
imgArrTiger = imgArrTiger[None]

imgWolf = image.load_img('testdata/wolf.jpg', target_size=(300, 300))
imgArrWolf = image.img_to_array(imgWolf)
imgArrWolf *= 1. / 255
imgArrWolf = imgArrWolf[None]

ans = ['кошка', 'собака']

predict = model.predict(imgArrCat)
print('Кот похож на: ', ans[np.argmax(predict)])
plt.imshow(imgCat)
plt.show()

predict = model.predict(imgArrDoge)
print('Собака похожа на: ', ans[np.argmax(predict)])
plt.imshow(imgDoge)
plt.show()

predict = model.predict(imgArrTiger)
print('Тигр похож на: ', ans[np.argmax(predict)])
plt.imshow(imgTiger)
plt.show()

predict = model.predict(imgArrWolf)
print('Волк похож на: ', ans[np.argmax(predict)])
plt.imshow(imgWolf)
plt.show()