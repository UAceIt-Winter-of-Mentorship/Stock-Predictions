import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 

df = data.get_data_tiingo('GOOGL', api_key="14a6f948d2203e5110a13d65f0599d4360cca0f5" )
df=df.reset_index()
df=df.drop(['date','symbol','adjClose','adjLow','adjHigh','adjOpen','adjVolume','divCash','splitFactor'], axis=1)
#print(df.tail(10))
#plt.plot(df.close)
ma30=df.close.rolling(30).mean()
#print(ma30)
#plt.plot(df.close, "#e52165")
#plt.plot(ma30,'#0d1137')
#plt.show()
training = pd.DataFrame(df['close'][0:int(len(df)*0.7)]) 
testing = pd.DataFrame(df['close'][int(len(df)*0.7):int(len(df))])
#print(training.shape)
#print(testing.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
training_array = scaler.fit_transform(training)
#print(training_array)
x_train=[]
y_train=[]
for i in range(100,training_array.shape[0]):
  x_train.append(training_array[i-100:i])
  y_train.append(training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50,activation="relu", return_sequences=True, 
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation="relu",return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation="relu", return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation="relu"))
model.add(Dropout(0.5))

#this is output layer 
model.add(Dense(units=1))
#print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)
model.save('keras_model.h5')
print(testing.head())
print(training.tail())
past_100_days= training.tail(100)
final_df = past_100_days.append(testing, ignore_index=True)
#print(final_df.tail())
input_data = scaler.fit_transform(final_df)
#print(input_data.shape)
x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)
y_predicted = model.predict(x_test)
scale_factor=1/0.00052396
y_predicted= y_predicted* scale_factor
y_test = y_test * scale_factor

plt.plot(y_test)
plt.plot(y_predicted, ls="dashed")
plt.title("Google stock price")
plt.xlabel("Days")
plt.ylabel("Price");
plt.grid()
plt.show()




