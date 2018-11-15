alphabet = ['A', 'T', 'C', 'G']
positive = True # keeps track of whether next sequence is pos or neg
datafile = 'single_species_big.txt'
pos_file = 'cami.pos'
neg_file = 'cami.neg'

with open(datafile, 'r') as data, open(pos_file, 'w') as pos, open(neg_file, 'w') as neg:
    # this won't work if file cannot fit into memory
    for line in list(data)[1:]: # iterate through backwards since pos and neg labels occur after each sequence
        if line[len(line) - 2] == '0': # add to neg
            positive = False
        elif line[len(line) - 2] == '1': # add to pos
            positive = True
        if line[0] in alphabet:
            if positive:
                pos.write(' '.join(line[:len(line) - 3]) + '\n')
            else:
                neg.write(' '.join(line[:len(line) - 3]) + '\n')
