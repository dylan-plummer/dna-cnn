import os
import time
import data_helpers as dhrt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, BatchNormalization, Input, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

# reproducible research :)
np.random.seed(42)

# Disable Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Network Parameters
learning_rate = 0.001
batch_size = 256
nb_epoch = 15
steps_per_epoch = 10
num_classes = 2


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


def train_model(filename, word_length=10, verbose=False):
    # load data
    dir = os.getcwd() + '/histone_data/'

    x_rt, y_rt = dhrt.load_data_and_labels(dir + 'pos/' + filename + '.pos', dir + 'neg/' + filename + '.neg')
    x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
    y_rt = np.array(list(y_rt))
    shuffled_rt = np.random.permutation(range(len(x_rt)))
    x_shuffle = x_rt[shuffled_rt]
    y_shuffle = y_rt[shuffled_rt]

    x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                          y_shuffle,
                                                          stratify=y_shuffle,
                                                          test_size=0.2)

    encoder = Input(shape=(None, 4))

    output = Conv1D(128, word_length) (encoder)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(5)(output)
    output = Conv1D(256, 2) (output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(20)(output)
    output = Dense(64, activation='relu')(output)
    output = SpatialDropout1D(0.6)(output)
    output = GlobalMaxPooling1D()(output)
    output = Dense(num_classes, activation='softmax')(output)
    model = Model(inputs=encoder,
                  outputs=output)

    adam = Adam(lr=learning_rate)
    sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['categorical_accuracy'])
    if verbose:
        print('Training shapes:', x_train.shape, y_train.shape)
        print('Valid shapes:', x_valid.shape, y_valid.shape)
        print(model.summary())
        #plot_model(model, to_file='model.png', show_shapes=True)

    history = model.fit_generator(generate_profile_batch(x_train, y_train),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=nb_epoch * len(x_train)//batch_size//steps_per_epoch,
                                  validation_data=generate_profile_batch(x_valid, y_valid),
                                  validation_steps=steps_per_epoch,
                                  verbose=0)
    # Save the weights
    model.save_weights('models/' + filename + '/' + str(word_length) + '_model_weights.h5')

    # Save the model architecture
    with open('models/' + filename + '/' + str(word_length) + '_model_architecture.json', 'w') as f:
        f.write(model.to_json())


    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('models/' + filename + '/' + str(word_length) + '_acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['total_loss', 'total_val_loss'], loc='upper left')
    plt.savefig('models/' + filename + '/' + str(word_length) + '_loss.png')
    plt.clf()

    score = model.evaluate_generator(generate_profile_batch(x_rt, y_rt),
                                     verbose=0,
                                     steps=len(x_rt)//batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])
    return score[0], score[1]


datasets = ['h3k36me3', 'h3k79me3', 'h4', 'h4ac']
word_lengths = [4, 5, 10, 16, 32]
performance = {}

for filename in datasets:
    accuracies = []
    for word_len in word_lengths:
        print('Training model on', filename, 'with word length', word_len)
        start_time = time.time()
        loss, acc = train_model(filename, word_length=word_len, verbose=False)
        accuracies.append(acc)
        print('Training took', time.time() - start_time, 'seconds\n')
    performance[filename] = accuracies

df = pd.DataFrame.from_dict(performance, columns=['k=4', 'k=5', 'k=10', 'k=16', 'k=32'], orient='index')
print(df)
df.to_pickle('performance.pkl')
