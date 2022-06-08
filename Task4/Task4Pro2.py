import re
from sklearn import preprocessing  # Пакет предварительной обработки данных
import datetime as datetime
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization  # Основные слои
from tensorflow.keras.optimizers import Adam  # Подключаем оптимизатор Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt  # Отрисовка изображений
from tensorflow.keras.callbacks import LambdaCallback # подключаем колбэки
from tqdm import tqdm

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

flatsData = pd.read_csv('content/moscow.csv', sep=";")

# Выбираем нечётные строки, в чётных строках в исходном фрейме пустые строки для комментариев
flatsData = flatsData.iloc[::2, :]

pd.set_option('display.max_columns', None)
# hhdata = hhdata.drop(hhdata.columns[0], 1)

print(flatsData.shape)
print(flatsData.head())


def getDistance(data):
    if data != data:
        distance = 0
    else:
        try:
            distance = int(data[:-1])
        except:
            print(data)

    return distance

def getDistanceExist(data):
    if data != data:
        exist = 0
    else:
        exist = 1

    return exist


def getDistanceTransport(data):
    transport = 0
    if 'т' in data:
        transport = 1
    return transport


def getFloor(data):
    try:
        floor = re.findall('\A\d+', data)[0]
    except:
        floor = 0
    return int(floor)


def getAllFloors(data):
    try:
        floors = re.findall('(\d+) ', data)[0]
    except:
        floors = 0
    return int(floors)


def getHouseType(data):
    ht = re.findall(' (\D+)', data)[0]
    return ht


def getSquare(data):
    existSquare = 1
    sq = 0
    sq1 = 0
    sq2 = 0
    try:
        sq = float(re.findall('\A[0-9.]+', data)[0])
        sq1 = float(re.findall('\D[0-9.]+', data)[0])
        sq2 = float(re.findall('\D[0-9.]+', data)[1])
    except:
        existSquare = 0

    return [sq, sq1, sq2, existSquare]


def getPrice(data):
    return int(data)


def getAgentBonus(data):
    num = 0
    if 'руб' in str(data):
        num = re.findall('\A[0-9. ]+', data)[0][:-1]
        num = num.replace(" ", "")
    return int(num)


def getAgentBonusProcent(data):
    num = 0

    if '%' in str(data):
        num = re.findall('\A\d+', data)[0]
    if num != num:
        num = 0
    return float(num)


def getDataYear(data):
    year = re.findall('\d{4}', data)[0]
    return int(year)


def getDataMonth(data):
    month = re.findall('[.](\d+)[.]', data)[0]
    return int(month)


def getExpositions(data):
    if data != data:
        return [0, 0]
    else:
        return [1, int(data)]


def getDescription(value):
    description = []
    buff = ''
    for val in value:
        buff += str(val[13]) + ' '
        description.append(buff)
        buff = ''
    return description


def create_dict(st):
    dct = {}
    for id, name in enumerate(st):
        dct.update({name: id})
    return dct


def to_ohe(val, dct):
    arr = [0] * len(dct)
    arr[dct[val]] = 1
    return arr


roomsData = []
for item in tqdm(flatsData.values):
    roomsData.append(item[0])

metrosData = []
for item in tqdm(flatsData.values):
    metrosData.append(item[1])

distanceData = []
for item in tqdm(flatsData.values):
    distanceData.append(getDistance(item[2]))

distanceDataExist = []
for item in tqdm(flatsData.values):
    distanceDataExist.append(getDistance(item[2]))

floorData = []
for item in tqdm(flatsData.values):
    floorData.append(getFloor(item[3]))

allFloorsData = []
for item in tqdm(flatsData.values):
    allFloorsData.append(getAllFloors(item[3]))

houseTypeData = []
for item in tqdm(flatsData.values):
    houseTypeData.append(getHouseType(item[3]))

balcoonsData = []
for item in tqdm(flatsData.values):
    balcoonsData.append(item[4])

toiletsData = []
for item in tqdm(flatsData.values):
    toiletsData.append(item[5])

squareData = []
for item in tqdm(flatsData.values):
    squareData.append(getSquare(item[6]))

priceData = []
for item in tqdm(flatsData.values):
    priceData.append(getPrice(item[7]))

agentBonusData = []
for item in tqdm(flatsData.values):
    agentBonusData.append(getAgentBonus(item[9]))

agentBonusProcentData = []
for item in tqdm(flatsData.values):
    agentBonusProcentData.append(getAgentBonusProcent(item[9]))

yearData = []
for item in tqdm(flatsData.values):
    yearData.append(getDataYear(item[10]))

monthData = []
for item in tqdm(flatsData.values):
    monthData.append(getDataMonth(item[10]))

expData = []
for item in tqdm(flatsData.values):
    expData.append(getExpositions(item[11]))

streamData = []
for item in tqdm(flatsData.values):
    streamData.append(item[12])


descriptions = getDescription(flatsData.values)
tokenizer = Tokenizer(num_words=2000,
                      filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0',
                      lower=True,
                      split=' ',
                      oov_token='unknown',
                      char_level=False)

tokenizer.fit_on_texts(descriptions)
descriptionsIndexes = tokenizer.texts_to_sequences(descriptions)
descriptionsIndexesMatrix = tokenizer.sequences_to_matrix(descriptionsIndexes)


distanceData = preprocessing.scale(distanceData)
squareData0 = preprocessing.scale([squareData[i][0] for i in range(len(squareData))])
squareData1 = preprocessing.scale([squareData[i][1] for i in range(len(squareData))])
squareData2 = preprocessing.scale([squareData[i][2] for i in range(len(squareData))])

squareDataExist = [squareData[i][3] for i in range(len(squareData))]
agentBonusData = preprocessing.scale(agentBonusData)
agentBonusProcentData = preprocessing.scale(agentBonusProcentData)
yearData = preprocessing.scale(yearData)
monthData = preprocessing.scale(monthData)
expDataExist = [expData[i][0] for i in range(len(expData))]

expData0 = []
for item in tqdm(expData):
    expData0.append(item[1])


x_train = []
roomsDatadct = create_dict(set(roomsData))
metrosDatadct = create_dict(set(metrosData))
floorDatadct = create_dict(set(floorData))
allFloorsDatadct = create_dict(set(allFloorsData))
houseTypeDatadct = create_dict(set(houseTypeData))
balcoonsDatadct = create_dict(set(balcoonsData))
toiletsDatadct = create_dict(set(toiletsData))
streamDatadct = create_dict(set(streamData))

for index, element in enumerate(tqdm(flatsData.values)):
    buf = []
    buf.extend(to_ohe(roomsData[index], roomsDatadct))
    buf.extend(to_ohe(metrosData[index], metrosDatadct))
    buf.extend(to_ohe(floorData[index], floorDatadct))
    buf.extend(to_ohe(allFloorsData[index], allFloorsDatadct))
    buf.extend(to_ohe(houseTypeData[index], houseTypeDatadct))
    buf.extend(to_ohe(balcoonsData[index], balcoonsDatadct))
    buf.extend(to_ohe(toiletsData[index], toiletsDatadct))
    buf.extend(to_ohe(streamData[index], streamDatadct))
    buf.append(distanceData[index])
    buf.append(squareData0[index])
    buf.append(squareData1[index])
    buf.append(squareData2[index])
    buf.append(squareDataExist[index])
    buf.append(agentBonusData[index])
    buf.append(agentBonusProcentData[index])
    buf.append(yearData[index])
    buf.append(monthData[index])
    buf.append(expDataExist[index])
    buf.append(expData0[index])
    buf.extend(descriptionsIndexesMatrix[index])

    x_train.append(buf)

x_train = np.array(x_train, dtype=np.float64)
y_train = np.array(priceData, dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05)


sc = StandardScaler()
y_train = sc.fit_transform(y_train.reshape(-1, 1)).flatten()


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


def on_epoch_end(epoch, logs):
  pred = model.predict(x_val) #Полуаем выход сети на проверочно выборке
  predUnscaled = sc.inverse_transform(pred).flatten() #Делаем обратное нормирование выхода к изначальным величинам цен квартир
  yTrainUnscaled = sc.inverse_transform(y_val.reshape(-1, 1)).flatten() #Делаем такое же обратное нормирование yTrain к базовым ценам
  delta = predUnscaled - yTrainUnscaled #Считаем разность предсказания и правильных цен
  absDelta = abs(delta) #Берём модуль отклонения
  print(predUnscaled[:3], '   ', yTrainUnscaled[:3])
  print("Эпоха", epoch, "модуль ошибки", round(sum(absDelta) / (1e+6 * len(absDelta)),3)) #Выводим усреднённую ошибку в миллионах рублей

pltMae = LambdaCallback(on_epoch_end=on_epoch_end)

model = Sequential()
model.add(Dense(300, activation='relu', input_dim=(x_train.shape[1])))

model.add(Dense(150, activation='relu'))

model.add(Dense(15, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    epochs=60,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    shuffle=True,
                    callbacks=[pltMae])

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
         label='mae на обучающем наборе')
plt.plot(history.history['val_mae'],
         label='mae на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('mae')
plt.legend()
plt.grid()
plt.show()


predict = model.predict(x_test)
predict_inverse = sc.inverse_transform(predict).flatten()

sumError = 0
for i in range(len(y_test)):
    buf = abs(y_test[i] - predict_inverse[i])
    sumError += buf
    if i < 10:
        print('Реальная цена: ', y_test[i], ', предугаданная цена: ', predict_inverse[i])

print(len(y_test))
print(sumError)
sumError /= len(y_test)

print('Средняя суммарная ошибка: ', round(sumError, 2), sep='')
# с нормализацией 11981356
# без нормализации 11322051
