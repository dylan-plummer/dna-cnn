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
    score = np.array(list(a)[2])
    #if np.array_equal(y[seq_i], y[seq_j]):
    align_y = np.array([y[seq_i], y[seq_j]])
    for i in range(seq_i + 1, seq_i + batch_size):
        for j in range(seq_j + 1, seq_j + batch_size):
            a = pairwise2.align.localxx(x[i], x[j], one_alignment_only=True)[0]
            align_x = np.vstack((align_x, np.array(list(a)[0:2])))
            score = np.append(score, list(a)[2])
            #if np.array_equal(y[i], y[j]):
            #    align_y = np.append(align_y, np.array([y[i]]))
            #else:
            align_y = np.append(align_y, np.array([y[i], y[j]]))
    lens = [len(seq) for alignment in align_x for seq in alignment]
    max_document_length = max(lens)
    align_y = np.reshape(align_y, (-1, 2))
    return align_x, align_y, score, max_document_length


def class_to_onehot(y, num_classes):
    onehot = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        onehot[i][y[i]] = 1
    return onehot


def split_alignments(x, max_len, word_length):
    s1 = []
    s2 = []
    for i in range(len(x)):
        proc1 = ''
        proc2 = ''
        for j in range(max_len):
            if j < len(x[i][0]):
                proc1 += (x[i][0][j]).replace('-', 'x')
                proc2 += (x[i][1][j]).replace('-', 'x')
            else:
                proc1 += 'x'
                proc2 += 'x'
        proc1 = ' '.join([proc1[i:i + word_length] for i in range(0, len(proc1))])
        proc2 = ' '.join([proc2[i:i + word_length] for i in range(0, len(proc2))])
        s1 = np.append(s1, proc1)
        s2 = np.append(s2, proc2)
    return s1, s2


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
    print(vocab)
    return vocab

print(pairwise2.align.localxx('aaatatatagcgacgactagc', 'acatcatctacgagcactatctagc', one_alignment_only=True))