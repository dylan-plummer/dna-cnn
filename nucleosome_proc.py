import os

alphabet = ['A', 'T', 'C', 'G']
positive = True # keeps track of whether next sequence is pos or neg
dataset = 'h4ac.txt'
datafile = os.getcwd() + '/histone_data/' + dataset
pos_file = os.getcwd() + '/histone_data/pos/' + dataset[:len(dataset) - 4] + '.pos'
neg_file = os.getcwd() + '/histone_data/neg/' + dataset[:len(dataset) - 4] + '.neg'

with open(datafile, 'r') as data, open(pos_file, 'w') as pos, open(neg_file, 'w') as neg:
    # this won't work if file cannot fit into memory
    for line in reversed(list(data)): # iterate through backwards since pos and neg labels occur after each sequence
        if line[0] == '0': # add to neg
            positive = False
        elif line[0] == '1': # add to pos
            positive = True
        if line[0] in alphabet:
            if positive:
                pos.write(" ".join(line))
            else:
                neg.write(" ".join(line))
