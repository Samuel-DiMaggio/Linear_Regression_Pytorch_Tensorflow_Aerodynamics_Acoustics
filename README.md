# Linear Regression Using Pytorch and Tensorflow on Aerodynamics & Acoustics Dataset

## Overview

The dataset contains a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel. The dataset contains the following features/predictors and label/target.

This dataset has the following features:

1. Frequency, in Hertz
2. The angle of attack, in degrees
3. Chord length, in meters
4. Free-stream velocity, in meters per second
5. Suction side displacement thickness, in meters

The only target/label is:

Scaled sound pressure level, in decibels

### Loading and cleaning up the dataset

```python
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("airfoil_self_noise.dat",sep='\t', header=None, skiprows=0,
                 low_memory = False, skipinitialspace=True,
                 names=['Frequency(Hertz)',
                        'The angle of attack (degrees)',
                        'Chord length (m)',
                        'Free-stream velocity (m/sec)',
                        'Suction side displac. thickness (m)',
                        'Scaled sound press. (decibels)'])
df.head()
```
![image](https://user-images.githubusercontent.com/47721595/152079833-85734e77-af6a-430e-a330-2e866846293a.png)

```python
df.dtypes
```
![image](https://user-images.githubusercontent.com/47721595/152079906-c63e5647-53d8-4f52-ae6a-cc4d52fc8ac9.png)

Next, let's check the missing values in the dataset.

```python
df.isnull().sum(axis=0)
```
![image](https://user-images.githubusercontent.com/47721595/152079965-c8084a0a-125b-4546-b9b2-6b58092d239d.png)

### Summarizing the dataset.

```python
df.describe()
```
![image](https://user-images.githubusercontent.com/47721595/152080026-90d77811-7e29-4b02-9593-8cdf58fc0e46.png)

```python
sns.pairplot(df)
```
![image](https://user-images.githubusercontent.com/47721595/152081397-1f83aec8-4d9e-474e-9b0f-b1e57e09315e.png)

```python
sns.boxplot(x='Free-stream velocity (m/sec)',y='Scaled sound press. (decibels)',data=df)
```
![image](https://user-images.githubusercontent.com/47721595/152081440-913c6e73-ee27-40b1-a1b0-924c54699308.png)

```python
sns.boxplot(x='Frequency(Hertz)',y='Scaled sound press. (decibels)',data=df)
sns.set(rc={'figure.figsize':(15,10)})
```
![image](https://user-images.githubusercontent.com/47721595/152081481-d4bfed5d-95da-4ae6-8864-16409da18e13.png)

### Building a simple linear regression to forecast "Scaled sound pressure level" using all other features and scikit-learn package.

```python
# Spliting the data into 80% for training and 20% for test
X = df.drop('Scaled sound press. (decibels)',axis=1)
y = df['Scaled sound press. (decibels)'].astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)

from sklearn.linear_model import LinearRegression
m = LinearRegression()
m = m.fit(X_train, y_train)
y_pred = m.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('The mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred)))
print('The root of mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred,squared = False)))
print('The mean absolute error is {0:.4f}'.format(mean_absolute_error(y_test,y_pred)))
```
![image](https://user-images.githubusercontent.com/47721595/152081562-022ebfff-f7c3-4b9c-99af-3cdbfd4f860f.png)

### Preprocess the data using the normalization method to convert all features into the range of [0,1].

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Building a deep learning regression model to forecast "Scaled sound pressure level" using all other features and TensorFlow. 

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',loss='mse')

tf.random.set_seed(1)
model.fit(x=X_train,y=y_train,batch_size=32,epochs=100,
          validation_data=(X_test,y_test))
```
![image](https://user-images.githubusercontent.com/47721595/152081668-089e93f3-f096-4bc3-b1b8-fe85c7ddc769.png)

```python
X_train.shape
```
![image](https://user-images.githubusercontent.com/47721595/152081721-3afe61c8-4784-4c69-ab79-3843b4f00616.png)


```python
y_pred2 = model.predict(X_test)

print('The mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred2)))
print('The root of mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred2,squared = False)))
print('The mean absolute error is {0:.4f}'.format(mean_absolute_error(y_test,y_pred2)))
```
![image](https://user-images.githubusercontent.com/47721595/152081762-b7085162-1abf-42f5-bb29-dfd335e26b1f.png)

### Improving the model performance by adjusting the number of neurons.

```python
model_test = keras.Sequential()
model_test.add(layers.Dense(4, activation='relu'))
model_test.add(layers.Dense(1))

model_test.compile(optimizer='adam',loss='mse')

tf.random.set_seed(1)
model_test.fit(x=X_train,y=y_train,batch_size=32,epochs=100,
          validation_data=(X_test,y_test))
```
![image](https://user-images.githubusercontent.com/47721595/152081853-68624eea-9b40-4c27-9874-1d5e807ce018.png)

```python
y_pred_test = model_test.predict(X_test)

print('The mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred_test)))
print('The root of mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred_test,squared = False)))
print('The mean absolute error is {0:.4f}'.format(mean_absolute_error(y_test,y_pred_test)))
```
![image](https://user-images.githubusercontent.com/47721595/152081896-aa8ddac5-d852-4be6-bb01-6b3ba623d487.png)


### Trying it with Pytorch 

```python
X_train = torch.tensor(X_train.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32).reshape(-1,1))

y_train.shape
```
![image](https://user-images.githubusercontent.com/47721595/152081992-bdbaab6a-3f93-4f30-a0db-37c7c0803ab4.png)

```python
input_size = X_train.shape[1]
output_size = y_train.shape[1]
hidden_size = 4
print(input_size)
print(output_size)
```
![image](https://user-images.githubusercontent.com/47721595/152082025-b0b58b04-99cd-4d53-ad7b-cd558fa437f2.png)

```python
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)  
        self.predict = torch.nn.Linear(hidden_size, output_size)  
    def forward(self, x):
        x = F.relu(self.hidden(x))     
        y_pred = self.predict(x)            
        return y_pred
 ```
 ```python
 model = LinearRegressionModel(input_size, hidden_size, output_size)
l = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

torch.manual_seed(1)
np.random.seed(0)
num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X_train.requires_grad_())
    loss= l(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {0}, loss:{1:.4f}'.format(epoch, loss.item()))
 ```
 ![image](https://user-images.githubusercontent.com/47721595/152082102-29148e2c-640d-48d3-9e83-023911943cf1.png)

```python
X_test = torch.from_numpy(X_test.astype(np.float32))
y_pred = model(X_test).detach().numpy()

print('The mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred)))
print('The root of mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred,squared = False)))
print('The mean absolute error is {0:.4f}'.format(mean_absolute_error(y_test,y_pred)))
```
![image](https://user-images.githubusercontent.com/47721595/152082156-808b106d-e697-44ed-b812-d815207f27cb.png)

### Improving the model performance by adjusting the number of neurons.

```python
input_size2 = X_train.shape[1]
output_size2 = y_train.shape[1]
hidden_size2 = 5
print(input_size)
print(output_size)
```
![image](https://user-images.githubusercontent.com/47721595/152082198-a2e50c5d-3d5a-4f23-a28d-b25f4f06ab04.png)

```python
class LinearRegressionModel2(torch.nn.Module):
    def __init__(self, input_size2, hidden_size2, output_size2):
        super(LinearRegressionModel2, self).__init__()
        self.hidden = torch.nn.Linear(input_size2, hidden_size2)  
        self.predict = torch.nn.Linear(hidden_size2, output_size2)  
    def forward(self, x):
        x = F.relu(self.hidden(x))     
        y_pred2 = self.predict(x)            
        return y_pred2
```
```python
model = LinearRegressionModel(input_size2, hidden_size2, output_size2)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.05)

torch.manual_seed(1)
np.random.seed(0)
num_epochs = 100
for epoch in range(num_epochs):
    y_pred2 = model(X_train.requires_grad_())
    loss= l(y_pred2, y_train)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    print('epoch {0}, loss:{1:.4f}'.format(epoch, loss.item()))
```
![image](https://user-images.githubusercontent.com/47721595/152082314-8b7cce6a-f598-468e-9475-9cb1c82de5b0.png)

```python
y_pred2 = model(X_test).detach().numpy()

print('The mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred2)))
print('The root of mean square error is {0:.4f}'.format(mean_squared_error(y_test,y_pred2,squared = False)))
print('The mean absolute error is {0:.4f}'.format(mean_absolute_error(y_test,y_pred2)))
```
![image](https://user-images.githubusercontent.com/47721595/152082359-e698c7c7-d89b-49ea-9a52-ce1b9d1d370e.png)

  
