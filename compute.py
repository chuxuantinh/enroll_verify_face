
import bob.bio.face
import numpy

####################################################################
### Some auxiliary functions

def hamming_distance(X, Y):
    '''Computes the noralised Hamming distance between two Bloom filter templates'''
    dist = 0

    N_BF = X.shape[0]
    for i in range(N_BF):
        A = X[i, :]
        B = Y[i, :]

        suma = sum(A) + sum(B)
        if suma > 0:
            dist += float(sum(A ^ B)) / float(suma)

    return dist / float(N_BF)

def getOriDistance(X,Y):
    return bob.math.chi_square(X.flatten(), Y.flatten())

def getBFDistance(X,Y):
    return hamming_distance(X,Y)