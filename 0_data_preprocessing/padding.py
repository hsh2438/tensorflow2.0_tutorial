import tensorflow as tf

inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [71, 1331, 4231]
]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post')
print(padded_inputs)
