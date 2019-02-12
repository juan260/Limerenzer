import scipy
import wave
import struct
import numpy as np
import os
import sys
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras
import itertools
from fourier import *

class corruptLibraryException(Exception):
    def __init__(self, value='The library seems to be corrupt'):
        self.value=value
    def __str__(self):
        return self.value

def getDirs():
    return next(os.walk('.'))[1]

def getFiles():
    return next(os.walk('.'))[2]


def checkLibrary(trainPairs):
    for pair in trainPairs:
        if(not(os.path.isfile(pair[0]) and os.path.isfile(pair[1]))):
            return False
    return True

def scanLibrary(path):
    actualPath = os.getcwd()
    os.chdir(path)
    models = getDirs()
    trainPairs=[]
    for model in models:
        os.chdir(model)
        instruments=getDirs()
        for instrument in instruments:
            os.chdir(instrument)
            audios=getFiles()
            for audio in audios:
                if audio[-5:]=='o.wav':
                    trainPairs.append(
                        [os.path.join(path, model, instrument, audio),
                        os.path.join(path,model, instrument, audio[:-5] + 'd.wav')])
            os.chdir('../')
        os.chdir('../')
    os.chdir(actualPath)
    if checkLibrary(trainPairs):
        return trainPairs
    else:
        print("Error: corrupted library")
        return None

def loadLibrary(path, division, parts=1, part=0):
    library = scanLibrary(path)
    library = library[part::parts]
    originCollection, oldSizeo = readWave(library[0][0], division)
    destCollection, oldSized = readWave(library[0][1], division)
    i=1
    for book in library[1:]:
        origin, sizeo= readWave(book[0], division)
        print('Iteration {0} of {1} with len {2}'.format(i,len(library), len(origin)))
        if sizeo != oldSizeo:
            raise corruptLibraryException('The sizes of two of the books don\'t seem to match')
        originCollection += origin
        dest, sized= readWave(book[1], division)
        if sized != oldSized:
            raise corruptLibraryException('The sizes of two of the books don\'t seem to match')
        destCollection += dest
        i+=1
        if not i%50:
            print('Successfully loaded {0} of {1}'.format(i, len(library)))
    return originCollection, destCollection, oldSizeo, oldSized

def loadBooks(books, division):
    dic = {}
    for book in books:
        if book[0] in dic:
            dic[book[0]].append(book[1])
        else:
            dic[book[0]] = [book[1]]

    for book in dic:
        values = dic[book]

        for div in values:
            pass


