import tensorflow as tf
print(tf.__version__)


# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE SHOULD END HERE
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # YOUR CODE SHOULD START HERE
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = myCallback()

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # YOUR CODE HERE
        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # YOUR CODE HERE
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # YOUR CODE HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks]  # YOUR CODE HERE
                        )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]

if __name__ == "__main__":
    train_mnist()
