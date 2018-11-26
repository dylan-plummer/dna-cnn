from itertools import product

import numpy as np
import gensim
from Bio import pairwise2
import os

from keras_preprocessing.text import Tokenizer

import data_helpers as dhrt
import matplotlib.pyplot as plt
from sequence_helpers import class_to_onehot
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Model
from sklearn.manifold import TSNE

word_length = 8
vec_length = 4

# load data
dir = os.getcwd() + '/histone_data/'
#x_rt, y_rt = dhrt.load_data_and_labels_pos(dir + 'pos/h3k4me1.pos', pos=1)
x1, y1 = dhrt.load_data_and_labels_pos(dir + 'pos/h3.pos', pos=0)
x2, y2 = dhrt.load_data_and_labels_pos(dir + 'pos/h4.pos', pos=8)

x_rt = np.concatenate((x1, x2))
y_rt = np.concatenate((y1, y2))

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
    print('Vocab:', vocab)
    return vocab



def replace_spaces(x):
    return x.replace(' ', '')


x_rt = np.array([replace_spaces(seq) for seq in x_rt])
y_rt = np.array(list(y_rt))

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


def alignment2vec(alignment, w2v):
    vec = []
    word_list = alignment.split(' ')
    if len(word_list[-1]) != word_length:
        word_list = word_list[:-1]
    for word in word_list:
        try:
            if len(vec) == 0:
                vec = w2v.word_vec(word)
            else:
                vec = np.vstack((vec, w2v.word_vec(word)))
        except Exception as e:
            print('Word', word, 'not in vocab')
    vec = np.reshape(vec, (len(word_list), -1))
    return vec


def get_test_alignment(x, seq_i, seq_j, tokenizer):
   # w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
    alignment = pairwise2.align.globalxx(x[seq_i], x[seq_j], one_alignment_only=True)[0]
    align_x = np.array(list(alignment)[0:2])
    s1 = align_x[0]
    s2 = align_x[1]
    s1 = [' '.join([s1[i:i + word_length] for i in range(0, len(s1))]).replace('-','x')]
    s2 = [' '.join([s2[i:i + word_length] for i in range(0, len(s2))]).replace('-', 'x')]
    s1 = np.array(tokenizer.texts_to_sequences(s1))
    s2 = np.array(tokenizer.texts_to_sequences(s2))
    return [s1, s2]

#w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)


tokenizer = Tokenizer()
vocab = get_vocab('atcgx')
tokenizer.fit_on_texts(vocab)

print(model.layers)
conv_embds = model.layers[2].get_weights()[0]
## Plotting function
def plot_words(data, start, stop, step):
    trace = plt.Scatter(
        x=data[start:stop:step,0],
        y=data[start:stop:step, 1],
        text=vocab.keys()[start:stop:step]
    )
    print('plot')
    plt.plot(trace)
    plt.show()

conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
plot_words(conv_tsne_embds, 0, 2000, 1)

'''
layer_outputs = [layer.output for layer in model.layers[2:]]
print(layer_outputs)
for layer in layer_outputs:
    print(layer)
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(get_test_alignment(x_valid, 0, 1, tokenizer))
#print(activations)
seq1 = np.argmax(activations[-3])
seq2 = np.argmax(activations[-2])
prediction = activations[-1]
print(seq1, seq2, prediction)
activation_1 = activations[10]
#print(activation_1)
for kernel in activation_1[0]:
    words = np.reshape(kernel, (-1, word_length))
    print('Learned alignment motifs:')
    print(np.array(tokenizer.sequences_to_texts(np.round(words))))

    #print(w2v.similar_by_vector(words[0], topn=1)[0])
    '''