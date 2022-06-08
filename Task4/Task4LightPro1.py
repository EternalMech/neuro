import re
from sklearn import preprocessing  # Пакет предварительной обработки данных
import datetime as datetime
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization # Основные слои
from tensorflow.keras.optimizers import Adam  # Подключаем оптимизатор Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt # Отрисовка изображений

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


hhdata = pd.read_csv('content/hh_fixed.csv')

pd.set_option('display.max_columns', None)
hhdata = hhdata.drop(hhdata.columns[0], 1)

print(hhdata.shape)
print(hhdata.head())


def getSex(data):
    sex = 0
    if data[0] == 'М':
        sex = 1
    return sex


def getAge(data):
    age = int(data[11:13])
    return age


def getSalary(data):
    num = data
    num = re.findall('\d+', num)
    num = num[0]

    value = data
    value = re.findall('[a-zA-Zа-яА-ЯёЁ]+', value)
    value = value[0]

    if value == 'USD':
        num = float(num) * 65
    elif value == 'KZT':
        num = float(num) * 0.17
    elif value == 'грн':
        num = float(num) * 2.6
    elif value == 'белруб':
        num = float(num) * 30.5
    elif value == 'EUR':
        num = float(num) * 70
    elif value == 'KGS':
        num = float(num) * 0.9
    elif value == 'сум':
        num = float(num) * 0.007
    elif value == 'AZN':
        num = float(num) * 37.5

    num = int(num)
    return num


def getDescription(value):
    description = []
    buff = ''
    for val in value:
        if type(val[2]) != float:
            buff += val[2] + ' '
        if type(val[7]) != float:
            buff += val[7] + ' '
        description.append(buff)
        buff = ''
    return description


def getCity(data):

    city = re.findall('\A[а-яА-ЯЁё]+', data)

    return city[0]

def getBusy(data):
    busy = [0] * 4
    if 'полная занятость' in data:
        busy[0] = 1
    if 'частичная занятость' in data:
        busy[1] = 1
    if 'стажировка' in data:
        busy[2] = 1
    if 'проектная работа' in data:
        busy[3] = 1
    return busy


def getWorkTime(data):
    workTime = [0] * 4
    if 'полный график' in data:
        workTime[0] = 1
    if 'гибкий график' in data:
        workTime[1] = 1
    if 'сменный график' in data:
        workTime[2] = 1
    if 'удаленная работа' in data:
        workTime[3] = 1

    return workTime


def getExpirience(data):
    year = 0
    month = 0

    try:
        year = int(re.findall('([0-9]+) (года|лет|год)', data)[0][0])
    except:
        print('no data')

    try:
        month = int(re.findall('([0-9]+) (месяц|месяца|месяцев)', data)[0][0])
    except:
        print('no data')

    if year != 0:
        year *= 12

    return year + month


def getEducation(data):
    higherEducation = 0
    try:
        if 'Высшее образование' in data:
            higherEducation = 1
    except:
        print(data)

    return higherEducation


# def updatedResume(data):
#     dt = datetime.strptime(data, '%Y-%m-%d %H:%M')
#     d = datetime.timestamp() - dt
#     datetime.timedelta()

def getAuto(data):
    if 'Имеется собственный автомобиль' in data:
        Auto = 1
    else:
        Auto = 0

    return Auto


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

a = []
for i in range(len(hhdata.values)):
    if getSalary(hhdata.values[i][1]) <= 10000:
        a.append(i)

hhdata = hhdata.drop(hhdata.index[a])

sexData = [getSex(hhdata.values[i][0]) for i in range(len(hhdata.values))]
ageData = [getAge(hhdata.values[i][0]) for i in range(len(hhdata.values))]

salaryData = [getSalary(hhdata.values[i][1]) for i in range(len(hhdata.values))]

descriptions = getDescription(hhdata.values)
tokenizer = Tokenizer(num_words=10000,
                      filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0',
                      lower=True,
                      split=' ',
                      oov_token='unknown',
                      char_level=False)

tokenizer.fit_on_texts(descriptions)
descriptionsIndexes = tokenizer.texts_to_sequences(descriptions)
descriptionsIndexesMatrix = tokenizer.sequences_to_matrix(descriptionsIndexes)


cityData = [getCity(hhdata.values[i][3]) for i in range(len(hhdata.values))]
busyData = [getBusy(hhdata.values[i][4]) for i in range(len(hhdata.values))]
worktimeData = [getWorkTime(hhdata.values[i][5]) for i in range(len(hhdata.values))]
experienceData = [getExpirience(hhdata.values[i][6]) for i in range(len(hhdata.values))]
educationData = [getEducation(hhdata.values[i][7]) for i in range(len(hhdata.values))]
autoData = [getEducation(hhdata.values[i][11]) for i in range(len(hhdata.values))]

ageData = preprocessing.scale(ageData)
experienceData = preprocessing.scale(experienceData)

x_train = []
for index in range(len(hhdata.values)):
    buf = []
    buf.extend(to_ohe(cityData[index], create_dict(set(cityData))))
    buf.append(sexData[index])
    buf.append(ageData[index])
    buf.extend(descriptionsIndexesMatrix[index])
    buf.extend(busyData[index])
    buf.extend(worktimeData[index])
    buf.append(experienceData[index])
    buf.append(educationData[index])
    buf.append(autoData[index])

    # buf = to_ohe(cityData[index], create_dict(set(cityData))) + [sexData[index]] + [ageData[index]] \
    #       + descriptionsIndexesMatrix[index] + busyData[index] + worktimeData[index] + [experienceData[index]] \
    #       + [educationData[index]] + [autoData[index]]

    x_train.append(buf)


# print(x_train[0])
x_train = np.array(x_train, dtype=np.float64)
y_train = np.array(salaryData, dtype=np.float64)


# print(x_train[0], y_train[0])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05)

sc = StandardScaler()
y_train = sc.fit_transform(y_train.reshape(-1, 1)).flatten()

model = Sequential()
BatchNormalization(input_dim=(x_train.shape[1]))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    epochs=60,
                    validation_split=0.15,
                    verbose=1,
                    shuffle=True)

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
plt.plot(history.history['mae'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


predict = model.predict(x_test)
predict_inverse = sc.inverse_transform(predict).flatten()


sumError = 0
for i in range(len(y_test)):
    buf = abs(y_test[i] - predict_inverse[i]) / y_test[i]
    sumError += buf
    if i < 10:
        print('Реальная ЗП: ', y_test[i], ', предугаданная зп: ', predict_inverse[i])

print(len(y_test))
print(sumError)
sumError /= len(y_test)

print('Средний суммарный процент ошибки: ', round(sumError*100, 2), "%", sep='')
