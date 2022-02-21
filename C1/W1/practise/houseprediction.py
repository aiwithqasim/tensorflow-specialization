import tensorflow as tf
import numpy as np
from tensorflow import keras


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array ([1, 2, 3, 4, 5, 6, 7], dtype=float)  # Your Code Here#
    ys = np.array([100, 150, 200, 250, 300, 350, 400], dtype=float)  # Your Code Here#
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])  # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')  # Your Code Here
    model.fit(xs, ys, epochs=1000)  # Your Code here
    return model.predict(y_new)[0]


if __name__ == '__main__':
    prediction = house_model([7.0])
    print(prediction)
