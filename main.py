import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

x = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')

x_train = x[:int(0.8 * x.shape[0])]
y_train = y[:int(0.8 * y.shape[0])]

x_test = x[int(0.8 * x.shape[0]):]
y_test = y[int(0.8 * y.shape[0]):]

model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='softmax'))
model.add(Dense(5, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=128)

print(y_test - model.predict(x_test))





