import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

x = pd.read_csv('features.csv', dtype='float')
y = pd.read_csv('labels.csv', dtype='float')
del y[y.columns[0]]

x_train = x[:int(0.8 * x.shape[0])]
y_train = y[:int(0.8 * x.shape[0])]

x_test = x[int(0.8 * x.shape[0]):]
y_test = y[int(0.8 * x.shape[0]):]

model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(y.shape[1], activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)
model.save('model.h5')

df = pd.DataFrame(model.predict(x_test))
df.to_csv('prediction.csv')
y_test.to_csv('y_true.csv')
df = y_test - df
df.to_csv('result.csv')
accuracy = []
for column in df.columns:
    accuracy.append(sum(df[column])/max(df[column]))
print(accuracy)





