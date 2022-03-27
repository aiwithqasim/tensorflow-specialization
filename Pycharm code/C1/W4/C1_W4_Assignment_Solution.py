import tensorflow as tf
import os
import zipfile

zip_ref = zipfile.ZipFile("./happy-or-sad.zip", 'r')
zip_ref.extractall("./h-or-s")
zip_ref.close()


# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):  # YOUR CODE HERE):

        # YOUR CODE START HERE

        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > DESIRED_ACCURACY:
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

        # YOUR CODE END HERE

    callbacks = myCallback()

    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your
    # implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # YOUR CODE HERE,
        tf.keras.layers.MaxPooling2D(2, 2),  # YOUR CODE HERE,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # YOUR CODE HERE,
        tf.keras.layers.MaxPooling2D(2, 2),  # YOUR CODE HERE,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # YOUR CODE HERE
        tf.keras.layers.MaxPooling2D(2, 2),  # YOUR CODE HERE
        tf.keras.layers.Flatten(),  # YOUR CODE HERE
        tf.keras.layers.Dense(512, activation='relu'),  # YOUR CODE HERE
        tf.keras.layers.Dense(1, activation='sigmoid')  # YOUR CODE HERE
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',  # YOUR CODE HERE,
                  optimizer=RMSprop(learning_rate=0.001),  # YOUR CODE HERE,
                  metrics=['accuracy'])  # YOUR CODE HERE)

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1 / 255)  # YOUR CODE HERE

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory("./h-or-s",  # YOUR CODE HERE
                                                        target_size=(150, 150),  # YOUR CODE HERE
                                                        batch_size=10,  # YOUR CODE HERE
                                                        class_mode='binary')  # YOUR CODE HERE
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit(train_generator,  # YOUR CODE HERE,
                        steps_per_epoch=8,  # YOUR CODE HERE,
                        epochs=15,  # YOUR CODE HERE,
                        verbose=1,  # YOUR CODE HERE,
                        callbacks=[callbacks]  # YOUR CODE HERE,
                        )

    return history.history['accuracy'][-1]


if __name__ == "__main__":
    train_happy_sad_model()