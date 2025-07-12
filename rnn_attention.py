import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W = tf.keras.layers.Dense(1)

    def call(self, inputs, return_attention=False):
        score = self.W(inputs)  # shape: (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)  # softmax over time axis
        context = tf.reduce_sum(weights * inputs, axis=1)  # shape: (batch, hidden)
        if return_attention:
            return context, weights
        return context

input_layer = tf.keras.Input(shape=(max_len,))
embedding = tf.keras.layers.Embedding(vocab_size, 128)(input_layer)
lstm = tf.keras.layers.LSTM(64, return_sequences=True)(embedding)
attention_output = Attention()(lstm)
output = tf.keras.layers.Dense(1, activation='sigmoid')(attention_output)

model = tf.keras.Model(input_layer, output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=15, validation_data=(x_test, y_test))

embedding = tf.keras.layers.Embedding(vocab_size, 128)(input_layer)
lstm = tf.keras.layers.LSTM(64, return_sequences=True)(embedding)
attention_layer = Attention()
context, attention_weights = attention_layer(lstm, return_attention=True)
visual_model = tf.keras.Model(input_layer, [context, attention_weights])

sample = x_test[0:1]
context_output, attn_weights = visual_model.predict(sample)

word_index = imdb.get_word_index()
reverse_words = {v: k for k, v in word_index.items()}
decoded = [reverse_words.get(i - 3, '?') for i in sample[0] if i >= 3]

plt.figure(figsize=(16, 3))
sns.heatmap([attn_weights[0][:len(decoded)].squeeze()], cmap='viridis', xticklabels=decoded)
plt.title('RNN + Attention Heatmap')
plt.xlabel('Words')
plt.yticks([])
plt.show()
