from itertools import product

import math
import numpy as np
from Bio import pairwise2
import draw_logo
import pandas as pd
import os

import data_helpers as dhrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Model
from keras.utils import np_utils

word_length = 10
vec_length = 4
num_classes = 2
batch_size = 128
sequences_per_family = -1

# load data
dir = os.getcwd() + '/histone_data/'

class_labels = ['h3', 'h3k4me1', 'h3k4me2', 'h3k4me3', 'h3k9ac', 'h3k14ac', 'h3k36me3', 'h3k79me3', 'h4', 'h4ac']

x_rt, y_rt = dhrt.load_data_and_labels(dir + 'pos/h3.pos', dir + 'neg/h3.neg')

x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
y_rt = np.array(list(y_rt))


def get_vocab(chars):
    vocab = {}
    i = 0
    words = product(chars, repeat=word_length)
    word_list = []  # Create a empty list
    for permutation in words:
        word_list.append(''.join(permutation))  # Join alphabet together and append in namList
    for word in word_list:
        vocab[word] = i
        i += 1
    return vocab


print('Test alignment:')
print(pairwise2.align.globalxx(x_rt[0], x_rt[1], one_alignment_only=True)[0:2])
print(y_rt[0], y_rt[1])

x_train, x_valid, y_train, y_valid = train_test_split(x_rt,
                                                      y_rt,
                                                      stratify=y_rt,
                                                      test_size=0.2)

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')


def profile_to_sequence(profile):
    vocab = ['a','c','g','t']
    sequence = ''
    for bp in profile:
        sequence += vocab[int(bp.argmax())]
    return sequence


def sequence_to_profile(seq):
    profile = []
    vocab = {'a':0,'c':1,'g':2,'t':3}
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
            x_batch = sequences_to_profile(x[i:i + batch_size])
            y_batch = np_utils.to_categorical(y, num_classes=num_classes)
            yield x_batch , y_batch[i:i + batch_size], class_labels


def generate_batch(x, y, tokenizer):
    while True:
        for i in range(len(x) - batch_size):
            x_seq = split_sequence(x[i:i + batch_size], word_length)
            x_batch = np.array(tokenizer.texts_to_sequences(x_seq))
            x_batch = np.reshape(x_batch, (x_seq.shape[0], -1))
            y_batch = np_utils.to_categorical(y, num_classes=num_classes)
            yield x_batch, y_batch[i:i + batch_size], class_labels


def plot_activations(units):
    r = 8
    c = 8
    plot_i = 0
    fig, axes = plt.subplots(r, c)
    for kernel in units:
        print(plot_i)
        i = plot_i // c
        j = plot_i % c
        kernel[kernel < 0 ] *= -1
        L = draw_logo.logo(kernel, name="P53")
        L.draw(ax=axes[i][j])
        motif = profile_to_sequence(kernel)
        axes[i][j].set_xlabel(motif, fontsize=8)
        axes[i][j].set_ylabel('')
        axes[i][j].set_title('')
        plot_i += 1
    #plt.tight_layout()
    plt.show()


def count_frequencies(motifs, x, y, label):
    frequencies = {}
    for motif in motifs:
        count = 0
        for i in range(len(x)):  # for every sequence in the dataset
            if y[i] == label:
                for j in range(len(x[i]) - word_length):  # for every index of the sequence
                    if x[i][j:j + word_length] == motif:
                        count += 1
        frequencies[motif] = count
    return frequencies


def plot_conv_layer(x, y):
    layer_outputs = [layer.output for layer in model.layers]
    print(layer_outputs)
    for layer in layer_outputs:
        print(layer)
    conv_embds = np.array(model.layers[1].get_weights())
    weights = conv_embds[0]
    weights = np.reshape(weights, (64, word_length, 4))
    motifs = []
    for kernel in weights:
        kernel[kernel < 0] *= -1
        motifs = np.append(motifs, profile_to_sequence(kernel))
    plot_activations(weights)
    positive_features = count_frequencies(motifs, x, y, 1)
    positive_table = pd.DataFrame.from_dict(positive_features, columns=['count'], orient='index')
    print('Most informative positive features:\n')
    print(positive_table.sort_values('count', ascending=False).head(9))

    negative_features = count_frequencies(motifs, x, y, 0)
    negative_table = pd.DataFrame.from_dict(negative_features, columns=['count'], orient='index')
    print('\n\nMost informative negative features:\n')
    print(negative_table.sort_values('count', ascending=False).head(9))


plot_conv_layer(x_rt, y_rt)