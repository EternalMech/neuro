import json       # Для сохранения данных
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing  # Пакет предварительной обработки данных
import numpy as np
import matplotlib.pyplot as plt # Отрисовка изображений
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization # Основные слои
from tensorflow.keras.optimizers import Adam  # Подключаем оптимизатор Adam
from sklearn.model_selection import train_test_split


# Открываем файл, где мы ранее запарсили данные с авто.ру
with open('/content/data.json', 'r') as json_file:
    data = json.load(json_file)

# Формируем нужные нам данные
markAuto = [data[i]['vehicle_info']['mark_info']['name'] for i in range(len(data))]
modelAuto = [data[i]['vehicle_info']['model_info']['name'] for i in range(len(data))]
yearAuto = [data[i]['documents']['year'] for i in range(len(data))]
mileageAuto = [data[i]['state']['mileage'] for i in range(len(data))]
bodyAuto = [data[i]['vehicle_info']['configuration']['body_type'] for i in range(len(data))]
transmissionAuto = [data[i]['vehicle_info']['tech_param']['transmission'] for i in range(len(data))]
fuelAuto = [data[i]['vehicle_info']['tech_param']['engine_type'] for i in range(len(data))]
powerAuto = [data[i]['vehicle_info']['tech_param']['power'] for i in range(len(data))]
fromSalonAuto = [data[i]['salon']['is_official'] for i in range(len(data))]
tamozhnyaClearAuto = [data[i]['documents']['custom_cleared'] for i in range(len(data))]
priceAuto = [data[i]['price_info']['RUR'] for i in range(len(data))]


# Создаем функции для преобразования данных в ohe
def create_dict(st):
    dct = {}
    for id, name in enumerate(st):
        dct.update({name: id})
    return dct

def to_ohe(val, dct):
    arr = [0] * len(dct)
    arr[dct[val]] = 1
    return arr


# Нормируем числовые показатели для нейронной сети
yearAuto = preprocessing.scale(yearAuto)
mileageAuto = preprocessing.scale(mileageAuto)
powerAuto = preprocessing.scale(powerAuto)
fromSalonAuto = [0 if (data[i]['salon']['is_official'] == False) else 1 for i in range(len(fromSalonAuto))]
tamozhnyaClearAuto = [0 if (data[i]['documents']['custom_cleared'] == False) else 1 for i in range(len(tamozhnyaClearAuto))]


# Формируем обучающую выборку и преобразуем текстовые значения в ohe
x_train = []
y_train = priceAuto

for index in range(len(data)):
    buf = to_ohe(markAuto[index], create_dict(set(markAuto))) + \
          to_ohe(modelAuto[index], create_dict(set(modelAuto))) + \
          to_ohe(bodyAuto[index], create_dict(set(bodyAuto))) + \
          to_ohe(transmissionAuto[index], create_dict(set(transmissionAuto))) + \
          to_ohe(fuelAuto[index], create_dict(set(fuelAuto))) + \
          [yearAuto[index]] + [mileageAuto[index]] + [powerAuto[index]] + \
          [fromSalonAuto[index]] + [tamozhnyaClearAuto[index]]

    x_train.append(buf)

# Преобразуем список в массив
x_train = np.array(x_train, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)

#Разделяем на тренировочную и тестовую выборку
x_train_n, x_test, y_train_n, y_test = train_test_split(x_train, y_train, test_size=0.05)

# Нормируем y_train_n
sc = StandardScaler()
y_train_scaled_n = sc.fit_transform(y_train_n.reshape(-1, 1)).flatten()

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(2000, input_dim=398, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# Обучаем нейронную сеть
history = model.fit(x_train_n,
                    y_train_scaled_n,
                    batch_size=64,
                    epochs=100,
                    validation_split=0.15,
                    verbose=1,
                    shuffle=True)

# Выводим графики обучения и корректируем нейронку для лучших результатов
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

# Подсчитываем ошибку на каждом примере тестовой выборки
predict = model.predict(x_test)
predict_inverse = sc.inverse_transform(predict).flatten()

# Выявляем суммарный процент ошибки и показываем реальную цену машины и предикт
# нейронки на первые 10 машин из тестовой выборки
sumError = 0
for i in range(len(y_test)):
    buf = abs(y_test[i] - predict_inverse[i]) / y_test[i]
    sumError += buf
    if i < 10:
      print('Цена машины: ', y_test[i], ', предугаданная цена: ', predict_inverse[i])

sumError /= len(y_test)

print('Средний суммарный процент ошибки: ', round(sumError*100, 2), "%", sep='')