import numpy as np
from itertools import product
from Bio import pairwise2

def get_alignments(x, y, seq_i, seq_j, batch_size):
    '''
    Aligns every pair of sequences to prepare input to CNN
    :param x: a set of sequences
    :param y: a set of labels
    :return: a set of pairwise alignments of the sets in x (cartesian product) x
    '''
    a = pairwise2.align.localxx(x[seq_i], x[seq_j], one_alignment_only=True)[0]
    align_x = np.array(list(a)[0:2])
    #if np.array_equal(y[seq_i], y[seq_j]):
    align_y = np.array([y[seq_i], y[seq_j]])
    for i in range(seq_i + 1, seq_i + batch_size):
        for j in range(seq_j + 1, seq_j + batch_size):
            a = pairwise2.align.localxx(x[i], x[j], one_alignment_only=True)[0]
            align_x = np.vstack((align_x, np.array(list(a)[0:2])))
            #if np.array_equal(y[i], y[j]):
            #    align_y = np.append(align_y, np.array([y[i]]))
            #else:
            align_y = np.append(align_y, np.array([y[i], y[j]]))
    lens = [len(seq) for alignment in align_x for seq in alignment]
    max_document_length = max(lens)
    align_y = np.reshape(align_y, (-1, 2))
    return align_x, align_y, max_document_length


def get_vocab(chars, word_length):
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