from itertools import product

import numpy as np
import gensim
from Bio import pairwise2
import os
import umap

from keras_preprocessing.text import Tokenizer

import data_helpers as dhrt
import matplotlib.pyplot as plt
from sequence_helpers import get_alignments, get_vocab, class_to_onehot, split_alignments
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Model
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

word_length = 8
vec_length = 4
num_classes = 2
batch_size = 4
sequences_per_family = 2000

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
                       #x0_neg, x1_neg, x2_neg, x3_neg, x4_neg, x5_neg, x6_neg, x7_neg, x8_neg, x9_neg))
y_rt = np.concatenate((y0, y1, y2, y3, y4, y5, y6, y7, y8, y9))

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
    print('Vocab:', vocab)
    return vocab


def plot_with_labels(low_dim_embs, labels, filename=None, text_alpha=0.8, **plot_kwargs):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    # plt.figure()  #in inches

    xx, yy = low_dim_embs[:, 0], low_dim_embs[:, 1]
    plt.scatter(xx, yy, **plot_kwargs)
    for ii, label in enumerate(labels):
        x, y = xx[ii], yy[ii]
        try:
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         size='small',
                         alpha=text_alpha,
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        except Exception as ex:
            pass
            # raise ex

    if filename is not None:
        plt.title(filename.split('.')[0])
        plt.savefig(filename)



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


def generate_batch(x, y, tokenizer, batch_size, iter):
    indices_i = np.arange(0, len(x) - 1 - batch_size)
    indices_j = np.arange(0, len(x) - 1 - batch_size)
    for _ in range(iter):
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        print(i, j, x.shape, y.shape)
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
        yield [s1, s2, score], align_y, score


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




tokenizer = Tokenizer()
vocab = get_vocab('atcgx')
tokenizer.fit_on_texts(vocab)
# Run predictions
if True:
    max_to_pred = 1000
    pred_res = np.zeros([max_to_pred, num_classes])
    act_res = np.zeros(max_to_pred)
    all_text = []
    all_titles = []
    pred_generator = generate_batch(x_train, y_train, tokenizer, batch_size=batch_size, iter=20)
    num_predded = 0
    for pred_inputs in pred_generator:
        X_pred, y_true, obj_title = pred_inputs
        # all_text += raw_text
        all_titles.append(obj_title)
        y_preds = model.predict(X_pred)

        offset = num_predded
        num_predded += X_pred[0].shape[0]

        pred_res[offset:offset + y_preds.shape[0], :] = y_preds
        act_res[offset:offset + y_true.shape[0]] = np.argmax(y_true, axis=0)

all_titles = np.array(all_titles)

conv_embds = model.layers[2].get_weights()[0]
print(conv_embds)


def plot_words(data, perplexity):
    center_points = np.zeros([num_classes, 2])

    color_map_name = 'gist_rainbow'
    cmap = plt.get_cmap(color_map_name)

    plt.figure()
    plt.hold(True)
    for cc in range(num_classes):
        # Plot each class using a different color
        cfloat = (cc + 1.0) / num_classes / 5
        keep_points = np.where(plot_act_res == cc)[0]
        cur_plot = data#[start:stop:step]

        cur_color = cmap(cfloat)
        # Label the final point, that's the Probability=1 point
        peak_label = '%s_tSNE' % cc

        # Scatter plot
        plt.plot(cur_plot[:, 0], cur_plot[:, 1], 'o', color=cur_color, alpha=0.5)

        x, y = cur_plot[-1, :]
        plt.annotate(peak_label,
                     xy=(x, y),
                     xytext=(5, 2),
                     size='small',
                     alpha=0.8,
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

        # Plot points in class 6 and 7, just to look at overlap
        # if cc in [6,7]:
        # plot_with_labels(cur_plot[0:-1], all_titles[keep_points[0:-1]], text_alpha=0.5)


    plt.title('tSNE Visualization. Perplexity %d' % perplexity)
    plt.legend(loc='lower right', numpoints=1, fontsize=6, framealpha=0.5)
    plt.show()


print('Fitting weights')
# Add class centers. Have to do this before the tSNE transformation
plot_data_points = np.concatenate([pred_res, np.identity(num_classes)], axis=0)
plot_act_res = np.concatenate([act_res, np.arange(num_classes)])
perplexity_list = [5, 30, 60, 250]

for perplexity in perplexity_list:
    embedding = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=5000, random_state=2157, verbose=1).fit_transform(plot_data_points)
    #embedding = umap.UMAP(n_neighbors=2,
                          #min_dist=0.1,
                          #metric='correlation',
                          #verbose=1).fit_transform(plot_data_points)
    print('Running viz')
    plot_words(embedding, perplexity)

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