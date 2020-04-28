"""
example of train mnist dataset with Tensorflow 2.0
"""

import tensorflow as tf


"""
load mnist dataset from tensorflow
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# divide with 255 to make pixels between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = tf.reshape(x_train, [-1, 28, 28, 1]), tf.reshape(x_test, [-1, 28, 28, 1])

"""
build model and set optimizer
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(28, 28, 1), dtype='float32', name='input'))
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
train
"""
model.fit(x_train, y_train, epochs=10)

"""
evaluate
"""
model.evaluate(x_test, y_test, verbose=2)
