from tensorflow.keras.datasets import cifar10  #Загружаем базу cifar10
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D       # Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam    # Подключаем оптимизатор Adam
from tensorflow.keras import utils              #Утилиты для to_categorical
import matplotlib.pyplot as plt  # Отрисовка графика
import optuna

#Загружаем cifar10
(x_train10, y_train10), (x_test10, y_test10) = cifar10.load_data()

y_train10 = utils.to_categorical(y_train10, 10)
y_test10 = utils.to_categorical(y_test10, 10)


# один слой 2 фильтров
model = Sequential()
model.add(Conv2D(2, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history0 = model.fit(x_train10,
                    y_train10,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# один слой 4 фильтра
model = Sequential()
model.add(Conv2D(4, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history1 = model.fit(x_train10,
                    y_train10,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# один слой 16 фильтров
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history2 = model.fit(x_train10,
                    y_train10,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# меняем активацию на linear
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='linear', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="linear"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history3 = model.fit(x_train10,
                    y_train10,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# меняем размер batch_size 10
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history4 = model.fit(x_train10,
                    y_train10,
                    batch_size=10,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# меняем размер batch_size 100
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history5 = model.fit(x_train10,
                    y_train10,
                    batch_size=100,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


# меняем размер batch_size 60000(вся база)
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history6 = model.fit(x_train10,
                    y_train10,
                    batch_size=60000,
                    epochs=30,
                    verbose=1,
                    validation_data = (x_test10, y_test10))


#Запишем точности нейронных сетей с различными гиперпараметрами

print('Один слой 2 фильтра: ', round(history0.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('Один слой 4 фильтров: ', round(history1.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('Один слой 16 фильтров ', round(history2.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('Активация relu -> linear: ', round(history3.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('batch_size 10: ', round(history4.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('batch_size 100: ', round(history5.history['val_accuracy'][-1]*100, 2), '%', sep = '')
print('batch_size 60000(вся база): ', round(history6.history['val_accuracy'][-1]*100, 2), '%', sep = '')


#Выводим графики обучения

print('Один слой 2 фильтра: ')

plt.plot(history0.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history0.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history0.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history0.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('Один слой 4 фильтра: ')

plt.plot(history1.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history1.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history1.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history1.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('Один слой 16 фильтра: ')

plt.plot(history2.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history2.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history2.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history2.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('Активация relu -> linear: ')

plt.plot(history3.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history3.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history3.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history3.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('batch_size 10: ')

plt.plot(history4.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history4.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history4.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history4.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('batch_size 100: ')

plt.plot(history5.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history5.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history5.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history5.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('batch_size 60000(вся база): ')

plt.plot(history6.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history6.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history6.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history6.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()