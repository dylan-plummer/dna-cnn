import os
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from Bio import pairwise2
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import callbacks
from keras.initializers import VarianceScaling
from sequence_helpers import get_vocab
from clustering_layer import ClusteringLayer
import data_helpers as dhrt

n_clusters = 10
word_length = 6
batch_size = 8
sequences_per_family = 1000
kmer_step = 3
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)

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

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


def tokenize_sequences(x_rt, tokenizer, word_length):
    x = np.array([])
    i = 0
    for seq in x_rt:
        words = [' '.join([seq[i:i + word_length] for i in range(0, len(seq), kmer_step)])]
        x = np.append(x, tokenizer.texts_to_sequences(words))
    return x

tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcg', word_length))
V = len(tokenizer.word_index) + 1
print('Num Words:', V)
print(x_rt)
#x_rt = np.array([seq.replace(' ', '') for seq in x_rt])
y = np.array(list(y_rt))
x = np.reshape(tokenize_sequences(x_rt, tokenizer, word_length), (x_rt.shape[0], -1))
print(x)
shuffled_rt = np.random.permutation(range(len(x)))

x_shuffle = x[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.2)
# Train K-Means for baseline
y_pred_kmeans = kmeans.fit_predict(x_train)
# Evaluate the K-Means clustering accuracy.
print(metrics.accuracy_score(y_train, y_pred_kmeans))

dims = [x.shape[-1], 2048, 2048, 5096, 10]
init = VarianceScaling(scale=1.0, mode='fan_in',
                           distribution='normal')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 2
autoencoder, encoder = autoencoder(dims, init=init)


autoencoder.compile(optimizer=pretrain_optimizer, loss='categorical_crossentropy')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.save_weights('ae_weights.h5')



autoencoder.load_weights('ae_weights.h5')

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer=SGD(0.01, 0.9), loss='categorical_hinge')
print(model.summary())

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])

tol = 0.0001 # tolerance threshold to stop training


for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.accuracy_score(y, y_pred), 5)
            #nmi = np.round(metrics.nmi(y_train, y_pred), 5)
            #ari = np.round(metrics.ari(y_train, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f' % (ite, acc), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights('DEC_model_final.h5')

#model.fit(x_train, y_train, batch_size=batch_size, epochs=4, validation_data=[x_valid, y_valid], verbose=1)