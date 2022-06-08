from tensorflow.keras.models import Sequential  # Сеть прямого распространения
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
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
import tensorflow as tf
import plotly.express as px
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

# Ограничиваем резервирование памяти для нейронок
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# SEED
seed = 256
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# CONFIGURATION
activation = 'relu'
inputShape = 300
dropout = 0.2
batch_size = 8
roughEpochs = 10
fineEpochs = 80

# коллбэки
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True, mode='max')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=1e-07, verbose=1)
checkpoint = ModelCheckpoint('outputs/simple_nn.h5',
                             monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

# # Загружаем датасет
# url = "https://drive.google.com/uc?id=12d4l1sgfXStBKsIEza2OjGvau06xN8ZK"
# output = "CatDogs.zip"
# gdown.download(url, output)

# with zipfile.ZipFile("CatDogs.zip","r") as zip_ref:
#     zip_ref.extractall()

# for activation in ['relu', 'sigmoid']:
#     for inputShape in [300, 200, 400, 500]:
#         for dropout in [0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]:
#             for batch_size in [8, 4, 128, 16, 32, 64]:
#                 for epochs in [100, 25, 30, 40]:



# Задаем параметры генератору изображений
datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
)

datagen1 = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
)

datagen2 = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
)

# Формируем тренировочные данные
train_generator = datagen.flow_from_directory(
    'samples/train',
    target_size=(inputShape, inputShape),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Формируем проверочные данные
val_generator = datagen1.flow_from_directory(
    'samples/val',
    target_size=(inputShape, inputShape),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

test_generator = datagen2.flow_from_directory(
    'samples/test',
    target_size=(inputShape, inputShape),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

fig, axs = plt.subplots(1, 2, figsize=(25, 5)) #Создаем полотно из 3 графиков
for i in range(2): #Проходим по всем классам
    car_path = 'samples/train' + '/' + os.listdir('samples/train')[i] + '/'#Формируем путь к выборке
    img_path = car_path + random.choice(os.listdir(car_path)) #Выбираем случайное фото для отображения
    axs[i].imshow(image.load_img(img_path, target_size=(inputShape, inputShape))) #Отображение фотографии
plt.show()
# Создаем модель нейронной сети


bmodel = InceptionV3(weights='imagenet', include_top=False)

x = bmodel.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=bmodel.input, outputs=predictions)


# Сперва тренируем только верхние слои(которые мы добавили)
# Т.е. заморозим развитие внутренних слоев импортируемой модели
for layer in bmodel.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=roughEpochs,
    verbose=1)

# Здесь верхние слои нейронки обучены и можно начать точное обучение (Fine tuning)

# for i, layer in enumerate(bmodel.layers):
#    print(i, layer.name)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True



# model = Sequential()

# model.add(Conv2D(32, (5, 5), input_shape=(inputShape, inputShape, 3), padding='same', activation=activation))
# model.add(Dropout(dropout))
# model.add(Conv2D(32, (5, 5), padding='same', activation=activation))
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Conv2D(64, (5, 5), padding='same', activation=activation))
# model.add(Dropout(dropout))
# model.add(Conv2D(64, (5, 5), padding='same', activation=activation))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(BatchNormalization())
#
# model.add(Conv2D(128, (5, 5), padding='same', activation=activation))
# model.add(Dropout(dropout))
# model.add(Conv2D(128, (5, 5), padding='same', activation=activation))
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Conv2D(256, (5, 5), padding='same', activation=activation))
# model.add(Dropout(dropout))
# model.add(Conv2D(256, (5, 5), padding='same', activation=activation))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
#
# model.add(Conv2D(512, (5, 5), padding='same', activation=activation))
# model.add(Dropout(dropout))
# model.add(Conv2D(512, (5, 5), padding='same', activation=activation))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Dropout(dropout))
# model.add(Flatten())
#
# model.add(Dense(512, activation=activation))
# model.add(Dropout(dropout))
# model.add(Dense(1024, activation=activation))
# model.add(Dropout(dropout))
# model.add(Dense(512, activation=activation))
# model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
#
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=val_generator,
#     validation_steps=val_generator.samples // batch_size,
#     epochs=epochs,
#     verbose=1)


# Выводим график ошибок
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения(грубый подбор)')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

# Выводим график точности
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения(грубый подбор)')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()

from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=fineEpochs,
    verbose=1)


# Выводим график ошибок
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения(тонкий подбор)')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

# Выводим график точности
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения(тонкий подбор)')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


results = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Тестовая ошибка:', round(results[0], 4))
print('Точность на тестовой выборке: ', round(results[1]*100, 2), '%', sep='')
accuracy = round(results[1]*100, 3)


pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.options.display.expand_frame_repr = False
# pd.set_option("precision", 2)

final_df = pd.read_excel('outputs/Task3UltraPro_georges_predict.xlsx')  # загружаем результаты анализа
# print(final_df)

data = {'activation': activation,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': fineEpochs,
        'input_shape': inputShape,
        'accuracy': accuracy}

final_df = final_df.append(data, ignore_index=True)

final_df.to_excel("outputs/Task3UltraPro_georges_predict.xlsx")

df_plot = final_df[['activation', 'dropout', 'batch_size', 'epochs', 'input_shape', 'accuracy']]

fig = px.parallel_coordinates(df_plot,
                              color="accuracy",
                              range_color=[df_plot['accuracy'].min(), df_plot['accuracy'].max()],
                              title='Зависимость точности от гиперпараметров нейронной сети',
                              color_continuous_scale=[
                                  (0.00, "gray"),   (0.75, "gray"),
                                  (0.75, "orange"), (1.00, "orange")
                              ])

fig.write_html("outputs/Task3UltraPro_georges_predict.html")  # сохраняем в файл
# fig.show()

