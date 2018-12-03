import random
import os
import data_helpers as dhrt
import numpy as np
from sequence_helpers import get_alignments, get_vocab, class_to_onehot, split_alignments

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Input, Reshape, GlobalMaxPooling1D, dot, multiply
from keras.layers import RepeatVector, concatenate, Permute
from keras.optimizers import SGD, Adam
from keras.layers.merge import Dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt
from Bio import pairwise2


# Network Parameters
learning_rate = 0.001
num_features = 372
word_length = 6
vec_length = 4
batch_size = 8
nb_epoch = 16
hidden_size = 25
sequences_per_family = 10
num_sequences = 10
steps_per_epoch = 4
num_classes = 20
num_filters = [16, 4]


def generate_batch(x, y, tokenizer):
    while True:
        for i in range(len(x) - batch_size):
            for j in range(len(y) - batch_size):
                if (i + batch_size) < len(x) and (j + batch_size) < len(x):
                    align_x, align_y, score, max_length = (get_alignments(x, y, i, j, batch_size))
                elif (i + batch_size) >= len(x):
                    print('End of training set, temp batch:', len(x[i:]))
                    align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[i:])))
                else:
                    print('End of training set, temp batch:', len(x[j:]))
                    align_x, align_y, score, max_length = (get_alignments(x, y, i, j, len(x[j:])))
                a1, a2 = split_alignments(align_x, max_length, word_length)
                s1, s2 = [], []
                for alignment in a1:
                    s1 = np.append(s1, alignment.split(' '))
                for alignment in a2:
                    s2 = np.append(s2, alignment.split(' '))
                s1 = np.array(tokenizer.texts_to_sequences(a1))
                s2 = np.array(tokenizer.texts_to_sequences(a2))
                s1 = np.reshape(s1, (a1.shape[0], -1))
                s2 = np.reshape(s2, (a2.shape[0], -1))
                y1 = np.array(np.hsplit(align_y, 2)[0].T[0])
                y2 = np.array(np.hsplit(align_y, 2)[1].T[0])
                align_y = 1*np.equal(y1, y2)
                #align_y = np_utils.to_categorical(align_y, num_classes=2)
                yield [s1, s2], align_y


# load data
dir = os.getcwd() + '/histone_data/'

x0, y0 = dhrt.load_data_and_labels_pos(dir + 'pos/h3.pos', pos=0, sequences_per_family=sequences_per_family)
x1, y1 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me1.pos', pos=1, sequences_per_family=sequences_per_family)
x2, y2 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me2.pos', pos=2, sequences_per_family=sequences_per_family)
x3, y3 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me3.pos', pos=3, sequences_per_family=sequences_per_family)
x4, y4 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k9ac.pos', pos=4, sequences_per_family=sequences_per_family)
x5, y5 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k14ac.pos', pos=5, sequences_per_family=sequences_per_family)
x6, y6 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k36me3.pos', pos=6, sequences_per_family=sequences_per_family)
x7, y7 = dhrt.load_data_and_labels_pos(dir + 'pos/h3k79me3.pos', pos=7, sequences_per_family=sequences_per_family)
x8, y8 = dhrt.load_data_and_labels_pos(dir + 'pos/h4.pos', pos=8, sequences_per_family=sequences_per_family)
x9, y9 = dhrt.load_data_and_labels_pos(dir + 'pos/h4ac.pos', pos=9, sequences_per_family=sequences_per_family)

x_rt = np.concatenate((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))
y_rt = np.concatenate((y0, y1, y2, y3, y4, y5, y6, y7, y8, y9))

x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
y_rt = np.array(list(y_rt))
shuffled_rt = np.random.permutation(range(len(x_rt)))
x_shuffle = x_rt[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]
print(pairwise2.align.localxx(x_shuffle[0], x_shuffle[1], one_alignment_only=True))

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.2)

print('x shape:', x_train.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcgx', word_length))
V = len(tokenizer.word_index) + 1
print('Num Words:', V)

alignment_batch = batch_size * batch_size - 2 * batch_size + 2
encoder_a = Input(shape=(None,))
layer_a = Embedding(V, hidden_size)(encoder_a)
#layer_a = Dropout(0.5)(layer_a)

encoder_b = Input(shape=(None,))
layer_b = Embedding(V, hidden_size)(encoder_b)
#layer_b = Dropout(0.5)(layer_b)

score = Input(shape=(1,))

decoder = concatenate([layer_a, layer_b], axis=1)

'''
output = Conv1D(128, word_length, activation='relu')(decoder)
output = MaxPooling1D(5)(output)
output = Conv1D(128, word_length, activation='relu')(decoder)
output = MaxPooling1D(20)(output)
'''

output = Bidirectional(LSTM(hidden_size))(decoder)
bias = concatenate([output, score], axis=1)
#output = Dense(32, activation='relu')(bias)
output = Dense(1, activation='sigmoid')(output)
model = Model(inputs=[encoder_a, encoder_b],
              outputs=output)

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['acc'])
print('Training shapes:', x_train.shape, y_train.shape)
print('Valid shapes:', x_valid.shape, y_valid.shape)
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit_generator(generate_batch(x_train, y_train, tokenizer),
                              steps_per_epoch=steps_per_epoch,
                              epochs=10 * len(x_train)//batch_size//steps_per_epoch,
                              validation_data=generate_batch(x_valid, y_valid, tokenizer),
                              validation_steps=steps_per_epoch)
# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['total_loss', 'total_val_loss'], loc='upper left')
plt.show()

