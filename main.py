import numpy as np
from models import Model
from layers import Dense
from optimizers import SGD

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[0],[1]])

model = Model()
model.add(Dense(n_dimension=4, n_inputs=2, activation='relu'))

optimizer = SGD(learning_rate=0.1)

model.compile(optimizer=optimizer, loss='mse')
print("Starting training...")

model.fit(x_train, y_train, epochs=10)
print("Training finished.")
