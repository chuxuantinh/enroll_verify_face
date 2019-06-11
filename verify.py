import bob.bio.face
import numpy
import os
import argparse
import math
from compute import *
import feature_extractor

parser = argparse.ArgumentParser()

parser.add_argument('username', help='username', type=str)

args = parser.parse_args()
username = args.username

def verifyOri(username):
    verify = numpy.loadtxt('db/verification/'+username+"_ori.txt").astype(int)
    for enrolled in os.listdir('db/enrollment/'+username+'/features/ori'):    
        
        enrolled = numpy.loadtxt('db/enrollment/'+username+'/features/ori/'+ enrolled[0:-4] + '.txt').astype(int)
        print("The original distance is: ",getOriDistance(enrolled,verify))

def verifyBF(username):
    verify = numpy.loadtxt('db/verification/'+username+"_bf.txt").astype(int)
    for enrolled in os.listdir('db/enrollment/'+username+'/features/bf'):    
        
        enrolled = numpy.loadtxt('db/enrollment/'+username+'/features/bf/'+ enrolled[0:-4] + '.txt').astype(int)
        print("The BF distance is: ",getBFDistance(enrolled,verify))

feature_extractor.extract_verify(username)
print("----------------------")
verifyOri(username)
print("----------------------")
verifyBF(username)