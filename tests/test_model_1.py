import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

# dummy data
X_train = np.random.rand(756, 30).astype('float32')
y_train = np.random.randint(0, 2, size=(756,)).astype('int64')

# model
model = Sequential([
    Dense(30, input_shape=(30,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

model.fit(X_train, y_train,
          validation_split=0.2,
          batch_size=8,
          epochs=1,
          shuffle=True,
          verbose=2)