import bob.io.base
import bob.bio.face
import numpy
import math
from PIL import Image
import os
import argparse
import json

# image resolution of the preprocessed images
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = 64

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 4, CROPPED_IMAGE_WIDTH / 4 * 3)

# Parameters of LGBPHS and Bloom filter extraction
N_BLOCKS = 80 # number of blocks the facial image is divided into, also for LGBPHS algorihtm

N_HIST = 40  # parameters fixed by LGBPHS
N_BINS = 59

THRESHOLD = 0  # binarization threshold for LGBPHS features

N_BITS_BF = 4  # parameters for BF extraction
N_WORDS_BF = 20
BF_SIZE = int(math.pow(2, N_BITS_BF))
N_BF_Y = N_HIST//N_BITS_BF
N_BF_X = (N_BINS+1)//N_WORDS_BF

# define facial LGBPHS feature extractor using bob face library
feature_extractor = bob.bio.face.extractor.LGBPHS(
    # block setup
    block_size = 8,
    block_overlap = 0,
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,
    # LBP setup (we use the defaults)
    # histogram setup
    sparse_histogram = False,
    split_histogram = 'blocks'
)

path = ''

####################################################################
### Some auxiliary functions

def extract_LGBPHS_features(filename,mode):
    global path
    tempPath = path+'/images/'
    if mode == "verify":
        tempPath = path
    '''Extracts unprotected template from image file'''
    image = bob.io.base.load(tempPath + filename[0:-4] + '.png')

    if image.ndim == 3:
        gray_image = bob.ip.color.rgb_to_gray(image)
    else:
        gray_image = image

    face_detector = bob.bio.face.preprocessor.FaceDetect(
        face_cropper='face-crop-eyes',
        use_flandmark=True,
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS}
    )

    tan_triggs_offset_preprocessor = bob.bio.face.preprocessor.TanTriggs(
        face_cropper = face_detector,
    )

    image2 = face_detector(gray_image)
    # bob.io.base.save(image2.astype('uint8'), grayDB + filename + '_cropped.png')
    gray_image = tan_triggs_offset_preprocessor(gray_image)
    # bob.io.base.save(gray_image.astype('uint8'), tanDB + filename + '_TanTriggs.png')
    return feature_extractor(gray_image)


def extract_BFs_from_LGBPHS_features(feat):
    '''Extracts BF protected template from an unprotected template'''
    template = numpy.zeros(shape=[N_BLOCKS * N_BF_X * N_BF_Y, BF_SIZE], dtype=int)

    index = 0
    for i in range(N_BLOCKS):
        block = feat[i, :]
        block = numpy.reshape(block, [N_HIST, N_BINS + 1])  # add column of 0s -> now done on features!

        block = (block > THRESHOLD).astype(int)

        for x in range(N_BF_X):
            for y in range(N_BF_Y):
                bf = numpy.zeros(shape=[BF_SIZE])

                ini_x = x * N_WORDS_BF
                fin_x = (x + 1) * N_WORDS_BF
                ini_y = y * N_BITS_BF
                fin_y = (y + 1) * N_BITS_BF
                new_hists = block[ini_y: fin_y, ini_x: fin_x]

                for k in range(N_WORDS_BF):
                    hist = new_hists[:, k]
                    location = int('0b' + ''.join([str(a) for a in hist]), 2)
                    bf[location] = int(1)

                template[index] = bf
                index += 1

    return template


####################################################################
### Template extraction

# Define permutation key to provide unlinkability
# key4 = numpy.zeros(shape=[N_BF_Y, N_BF_X, N_BITS_BF * N_BLOCKS], dtype=int)
# for j in range(N_BF_Y):
#     for k1 in range(N_BF_X):
#         key = numpy.random.permutation(N_BITS_BF * N_BLOCKS)
#         key4[j, k1, :] = key

# numpy.save("key4.npy", key4, allow_pickle=True, fix_imports=True)
key4=numpy.load("key4.npy", allow_pickle=True, fix_imports=True)
# print("trolllloollooloolol ",numpy.array_equal(key4,key3))

# print('shape key4 '+str(key4.shape))
# print(key4)

def main(username, mode):
    global path
    global key4

    if mode == 'enroll':
        path = 'db/enrollment/' + username
    elif mode == 'verify':
        path = 'db/verification/'
    else: 
        exit(0)

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+'/features'):
        os.mkdir(path+'/features')
    if not os.path.exists(path+'/features/ori'):
        os.mkdir(path+'/features/ori')
    if not os.path.exists(path+'/features/bf'):
        os.mkdir(path+'/features/bf')
    if not os.path.exists(path+'/images'):
        os.mkdir(path+'/images')


    print("Extracting LGBPHS features and BF templates for user ",username)
    for filename in os.listdir(path+'/images'):
            print("Begin processing file: ", filename)

            features = extract_LGBPHS_features(filename,mode)

            numpy.savetxt(path+'/features/ori/' + filename[0:-4] + '.txt', features, fmt='%d')


            features = numpy.reshape(features, newshape=[N_BLOCKS, N_HIST, N_BINS])

            feat = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1]) # to add a 0 at the end and round the N_BINS to 60
            feat[:, :, 0:N_BINS] = features

            # permute features to provide unlinkability
            features2 = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1])
            
            for j in range(N_BF_Y):
                for k1 in range(N_BF_X):
                    permKey = key4[j, k1, :]
                    aux = feat[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF]
                    aux = numpy.reshape(aux, [N_BLOCKS * N_BITS_BF, N_WORDS_BF])
                    aux = aux[permKey, :]
                    aux = numpy.reshape(aux, [N_BLOCKS, N_BITS_BF, N_WORDS_BF])
                    features2[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF] = aux

            # extract BFs
            bfs = extract_BFs_from_LGBPHS_features(features2)
            numpy.savetxt(path+'/features/bf/' + filename[0:-4] + '.txt', bfs, fmt='%d')
            print("Finish processing file: ", filename)

def extract_verify(username):
    global path
    global key4
    path = 'db/verification/'
    filename= username

    print("Extracting LGBPHS features and BF templates for user ",username)
    print("Begin processing file: ", filename)

    features = extract_LGBPHS_features(filename+".png","verify")

    numpy.savetxt(path + filename + '_ori.txt', features, fmt='%d')

    features = numpy.reshape(features, newshape=[N_BLOCKS, N_HIST, N_BINS])

    feat = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1]) # to add a 0 at the end and round the N_BINS to 60
    feat[:, :, 0:N_BINS] = features

    # permute features to provide unlinkability
    features2 = numpy.zeros([N_BLOCKS, N_HIST, N_BINS + 1])
    
    for j in range(N_BF_Y):
        for k1 in range(N_BF_X):
            permKey = key4[j, k1, :]
            aux = feat[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF]
            aux = numpy.reshape(aux, [N_BLOCKS * N_BITS_BF, N_WORDS_BF])
            aux = aux[permKey, :]
            aux = numpy.reshape(aux, [N_BLOCKS, N_BITS_BF, N_WORDS_BF])
            features2[:, j * N_BITS_BF: (j + 1) * N_BITS_BF, k1 * N_WORDS_BF: (k1 + 1) * N_WORDS_BF] = aux

    # extract BFs
    bfs = extract_BFs_from_LGBPHS_features(features2)
    numpy.savetxt(path+ filename + '_bf.txt', bfs, fmt='%d')
    print("Finish processing file: ", filename)


if __name__ == "__main__":
        ######################################################################
    ### Parameter and arguments definition
    parser = argparse.ArgumentParser()

    # location of source images, final templates and intermediate steps (the latter for debugging purposes)
    parser.add_argument('username', help='username', type=str)
    parser.add_argument('mode', help='mode', type=str)

    args = parser.parse_args()
    username = args.username
    mode = args.mode
    main(username,mode)