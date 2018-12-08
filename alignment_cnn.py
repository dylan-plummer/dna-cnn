import random
import os
import data_helpers as dhrt
import numpy as np
from sequence_helpers import get_alignments, get_vocab, class_to_onehot, split_alignments

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Input, Reshape, GlobalMaxPooling1D, dot, multiply
from keras.layers import RepeatVector, concatenate, Permute, SpatialDropout1D
from keras.optimizers import SGD, Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt
from Bio import pairwise2


# Network Parameters
learning_rate = 0.001
num_features = 372
word_length = 10
vec_length = 4
batch_size = 256
nb_epoch = 16
hidden_size = 100
sequences_per_family = -1
num_sequences = 10
steps_per_epoch = 10
num_classes = 2
num_filters = [16, 4]


def sequence_to_profile(seq):
    profile = []
    vocab = {'a':0,'t':1,'c':2,'g':3}
    for letter in seq:
        nuc = np.zeros(4)
        nuc[vocab[letter]] = 1
        profile = np.append(profile, nuc)
    profile = np.reshape(profile, (len(seq), 4))
    return profile


def sequences_to_profile(x):
    profile = np.array([])
    for seq in x:
        profile = np.append(profile, sequence_to_profile(seq))
    profile = np.reshape(profile, (x.shape[0], -1, 4))
    return profile


def split_sequence(x, word_length):
    split_seq = np.array([])
    for seq in x:
        split_seq = np.append(split_seq, ' '.join([seq[i:i + word_length] for i in range(0, len(seq))]))
    return split_seq


def generate_profile_batch(x, y):
    while True:
        for i in range(0, len(x) - batch_size - 1, batch_size):
            try:
                x_batch = sequences_to_profile(x[i:i + batch_size])
                y_batch = np_utils.to_categorical(y, num_classes=num_classes)
                yield x_batch , y_batch[i:i + batch_size]
            except ValueError as e:
                pass


def generate_batch(x, y, tokenizer):
    while True:
        for i in range(0, len(x) - batch_size - 1, batch_size):
            x_seq = split_sequence(x[i:i + batch_size], word_length)
            x_batch = np.array(tokenizer.texts_to_sequences(x_seq))
            x_batch = np.reshape(x_batch, (x_seq.shape[0], -1))
            y_batch = np_utils.to_categorical(y, num_classes=num_classes)
            yield x_batch, y_batch[i:i + batch_size]


# load data
dir = os.getcwd() + '/histone_data/'

x_rt, y_rt = dhrt.load_data_and_labels(dir + 'pos/h4.pos', dir + 'neg/h4.neg')

x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
y_rt = np.array(list(y_rt))
shuffled_rt = np.random.permutation(range(len(x_rt)))
x_shuffle = x_rt[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.2)

print('x shape:', x_train.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcg', word_length))
V = len(tokenizer.word_index) + 1
print('Num Words:', V)

alignment_batch = batch_size * batch_size - 2 * batch_size + 2
encoder = Input(shape=(None, 4))

output = Conv1D(64, word_length) (encoder)
#output = BatchNormalization()(output)
output = Activation('relu')(output)
output = AveragePooling1D(5)(output)
output = SpatialDropout1D(0.5)(output)
#output = AveragePooling1D(5)(output)
output = Conv1D(128, 2) (output)
output = Activation('relu')(output)
output = AveragePooling1D(20)(output)
output = SpatialDropout1D(0.5)(output)
#output = BatchNormalization()(output)
output = Dense(64, activation='relu')(output)
output = GlobalAveragePooling1D()(output)
output = Dense(num_classes, activation='softmax')(output)
model = Model(inputs=encoder,
              outputs=output)

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])
print('Training shapes:', x_train.shape, y_train.shape)
print('Valid shapes:', x_valid.shape, y_valid.shape)
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit_generator(generate_profile_batch(x_train, y_train),
                              steps_per_epoch=steps_per_epoch,
                              epochs=10 * len(x_train)//batch_size//steps_per_epoch,
                              validation_data=generate_profile_batch(x_valid, y_valid),
                              validation_steps=steps_per_epoch)
# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
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


print('Evaluating Model:')
score = model.evaluate_generator(generate_profile_batch(x_rt, y_rt),
                         verbose=1,
                         steps= len(x_rt)//batch_size)
print("Loss: ", score[0], "Accuracy: ", score[1])
