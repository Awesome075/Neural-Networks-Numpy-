import numpy as np
from models import Model
from layers import Dense
from optimizers import SGD
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=sns.load_dataset("penguins")
data.dropna(inplace=True)

le=LabelEncoder()
non_numerical_cols = ['species','island','sex']

for i in non_numerical_cols:
    data[i] = le.fit_transform(data[i])

X = data.drop('species' , axis=1)
y = data.species

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Model()
model.add(Dense(n_dimension=10, n_inputs=6, activation='relu'))
model.add(Dense(n_dimension=3, n_inputs=10, activation='softmax'))

optimizer = SGD(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='sparse_cross_entropy')

#X_train = X_train.values.astype(np.float32)       
y_train = y_train.values.astype(np.int32)
print("Starting training...")
model.fit(X_train, y_train, epochs=200)

print("Training finished.") 

print("Testing...")
#X_test = X_test.values.astype(np.float32)
y_test = y_test.values.astype(np.int32)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(y_test)
print(y_pred)
