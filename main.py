import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras.backend as K
import numpy as np
from matplotlib import pyplot as pt


x = pd.read_csv('features2.csv', dtype='float32')
y = pd.read_csv('labels2.csv', dtype='float32')
seed = 9
np.random.seed(seed)
new_index = np.arange(x.shape[0])
np.random.shuffle(new_index)
x = x.iloc[new_index].reset_index(drop=True)
y = y.iloc[new_index].reset_index(drop=True)
#0.8 * x.shape[0]
length = 0.8 * x.shape[0]
x_train = x[:int(length)]
y_train = y[:int(length)]

x_test = x[int(length):]
y_test = y[int(length):].reset_index(drop=True)


def custom_accuracy(y_true, y_pred):
    if K.equal(K.argmax(y_true), K.argmax(y_pred)):
        return K.constant(1)
    return K.constant(0)


model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=8)
model.save('model.h5')

prediction = model.predict(x_test)
pd.DataFrame(prediction).to_csv('prediction.csv')
y_test.to_csv('y_true.csv')

successfully_predicted = 0
for prediction_row, true_row in zip(prediction, y_test.values):
    if np.argmax(prediction_row) == np.argmax(true_row):
        successfully_predicted += 1

prediction = y_test.values - prediction
df = pd.DataFrame(prediction)
df.to_csv('result.csv', sep=';')
accuracy = []
for column in df.columns:
    accuracy.append(sum(df[column])/len(df[column]))
print(accuracy)

print(successfully_predicted / len(prediction))

pt.plot(history.history['categorical_accuracy'])
pt.show()

