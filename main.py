import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

x = pd.read_csv('features.csv', dtype='float32')
y = pd.read_csv('labels.csv', dtype='float32')
y.columns = [str(i) for i in range(y.shape[1])]

x_train = x[:int(0.8 * x.shape[0])]
y_train = y[:int(0.8 * y.shape[0])]

x_test = x[int(0.8 * x.shape[0]):]
y_test = y[int(0.8 * y.shape[0]):].reset_index(drop=True)

model = Sequential()
model.add(Dense(12, input_dim=x.shape[1], activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(y.shape[1], activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=128)
model.save('model.h5')

prediction = model.predict(x_test)
pd.DataFrame(prediction).to_csv('prediction.csv')
y_test.to_csv('y_true.csv')
prediction = y_test.as_matrix() - prediction
df = pd.DataFrame(prediction)
df.to_csv('result.csv')
accuracy = []
for column in df.columns:
    accuracy.append(sum(df[column])/len(df[column]))
print(accuracy)





