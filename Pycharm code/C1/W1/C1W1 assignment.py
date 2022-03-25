import tensorflow as tf
import numpy as np
from tensorflow import keras


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # alternate solution
    # xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    # ys = 0.5 * (xs + 1)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000)
    return model.predict(y_new)[0]

if __name__ == "__main__":

    prediction = house_model([7.0])
    print(prediction)