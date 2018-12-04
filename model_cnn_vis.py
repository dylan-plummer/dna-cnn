from itertools import product

import numpy as np
import gensim
from Bio import pairwise2
import os

from keras_preprocessing.text import Tokenizer

import data_helpers as dhrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sequence_helpers import get_alignments, get_vocab, class_to_onehot, split_alignments
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Model
from keras.utils import np_utils
from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE

word_length = 4
vec_length = 4
num_classes = 10
batch_size = 64
sequences_per_family = 1000

# load data
dir = os.getcwd() + '/histone_data/'

class_labels = ['h3', 'h3k4me1', 'h3k4me2', 'h3k4me3', 'h3k9ac', 'h3k14ac', 'h3k36me3', 'h3k79me3', 'h4', 'h4ac']

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

def split_sequence(x, word_length):
    split_seq = np.array([])
    for seq in x:
        split_seq = np.append(split_seq, ' '.join([seq[i:i + word_length] for i in range(0, len(seq))]))
    return split_seq


def generate_batch(x, y, tokenizer):
    while True:
        for i in range(len(x) - batch_size):
            x_seq = split_sequence(x[i:i + batch_size], word_length)
            x_batch = np.array(tokenizer.texts_to_sequences(x_seq))
            x_batch = np.reshape(x_batch, (x_seq.shape[0], -1))
            y_batch = np_utils.to_categorical(y, num_classes=num_classes)
            yield x_batch, y_batch[i:i + batch_size], class_labels



tokenizer = Tokenizer()
vocab = get_vocab('atcg')
tokenizer.fit_on_texts(vocab)
# Run predictions
if True:
    max_to_pred = 5000
    pred_res = np.zeros([max_to_pred, num_classes])
    act_res = np.zeros(max_to_pred)
    all_text = []
    all_titles = []
    pred_generator = generate_batch(x_train, y_train, tokenizer)
    num_predded = 0
    for pred_inputs in pred_generator:
        X_pred, y_true, obj_title = pred_inputs
        # all_text += raw_text
        all_titles.append(obj_title)
        y_preds = model.predict(X_pred)
        offset = num_predded
        num_predded += X_pred.shape[0]
        try:
            pred_res[offset:offset + y_preds.shape[0], :] = y_preds
            act_res[offset:offset + y_true.shape[0]] = np.argmax(y_true, axis=1)
        except:
            print('ooh fuk')
            break
        print(len(all_titles))
        if len(all_titles) == max_to_pred:
            break

all_titles = np.array(all_titles)

conv_embds = model.layers[1].get_weights()[0]
print(conv_embds)


def plot_words(data, actual, perplexity):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cc in range(num_classes):
        keep_points = np.where(plot_act_res == cc)[0]
        cur_plot = data[keep_points, :]
        point_labels = list(np.zeros_like(keep_points, dtype=str))
        point_labels[-1] = '%s_tSNE' % cc
        # Plot each class using a different color
        print('CC', cc)

        # Scatter plot
        if cc == 0:
            color = 'g'
            marker = '^'
        elif cc == 1:
            color = 'r'
            marker = 'o'
        elif cc == 2:
            color = 'b'
            marker = '*'
        elif cc == 3:
            color = 'c'
            marker = 'v'
        elif cc == 4:
            color = 'm'
            marker = '+'
        elif cc == 5:
            color = 'y'
            marker = 'x'
        elif cc == 6:
            color = '#00ff00'
            marker = 'D'
        elif cc == 7:
            color = '#ff9900'
            marker = '2'
        elif cc == 8:
            color = '#6600cc'
            marker = 'p'
        elif cc == 9:
            color = '#ff99ff'
            marker = 'X'

        ax.scatter(cur_plot[:, 0], cur_plot[:, 1], cur_plot[:, 2], c=color, marker=marker, alpha=0.3)

        # Plot the median of the points
        avg_label = class_labels[cc]
        low_dim_centers = np.median(cur_plot, axis=0)
        print(low_dim_centers)
        print(low_dim_centers.shape)
        ax.scatter(low_dim_centers[0], low_dim_centers[1], low_dim_centers[2], c=color, marker=marker, alpha=1.0, label=avg_label)
        ax.text(low_dim_centers[0], low_dim_centers[1], low_dim_centers[2], avg_label)

    plt.title('tSNE Visualization. Perplexity %d' % perplexity)
    plt.legend(loc='lower right', numpoints=1, fontsize=6, framealpha=0.5)
    plt.show()


print('Fitting weights')
# Add class centers. Have to do this before the tSNE transformation
plot_data_points = np.concatenate([pred_res, np.identity(num_classes)], axis=0)
plot_act_res = np.concatenate([act_res, np.arange(num_classes)])
perplexity_list = [30, 60, 250, 500]

for perplexity in perplexity_list:
    embedding = TSNE(perplexity=perplexity, n_components=3, init='random', n_iter=5000, random_state=2157, verbose=1).fit_transform(plot_data_points)
    #embedding = umap.UMAP(n_neighbors=2,
                          #min_dist=0.1,
                          #metric='correlation',
                          #verbose=1).fit_transform(plot_data_points)
    print('Running viz')
    plot_words(embedding, plot_act_res, perplexity)

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