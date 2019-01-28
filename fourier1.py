import scipy
import wave
import struct
import numpy as np
import os
import sys
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras

def chunkAndFFT(data, size):
    chunks = [data[x:x+size] for x in xrange(0, len(data), size)]
    for i in range(len(chunks)):
        chunks[i] = np.fft.fft(chunks[i], axis=0)
    return chunks

def readWave(path, division):

    rate, data = wavfile.read(path)
    if rate%division:
        print('No se puede dividir el rate ' + str(rate) +\
            ' entre ' + str(division))
        return None
    size = rate/division
    # Divide data into two separate channels and return
    channel1 = [el[0] for el in data]
    channel2 = [el[1] for el in data]
    # Divide into chunks
    fft1=chunkAndFFT(channel1, size)
    fft2=chunkAndFFT(channel2, size)
    return fft1, fft2, size

def writeWave(path, filtereddata, size, division):
    #TODO: reconstruir canales y 'chunks'
    filteredRealWrite = []
    for j in range(len(filtereddata)):
        filteredComplexWrite = np.fft.ifft(filtereddata,axis=0)

        filteredComplexWrite[0]=np.float64(-1)
        filteredRealWrite.append([np.int16(-1)]*len(filteredComplexWrite))
        for i in range(1, len(filteredComplexWrite)):
            filteredRealWrite[j][i]=np.int16(np.rint(\
	        np.real(np.sign(filteredComplexWrite[i]))*\
	        np.abs(filteredComplexWrite[i])))

        filteredRealWrite[j][i]=np.array(filteredRealWrite)
    toWrite=filteredRealWrite[0]
    for i in range(1, len(filteredRealWrite)):
        toWrite.append(filteredRealWrite[i])

    wavfile.write(path, division*size, toWrite)
    return toWrite

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


#data1, rate = readWave('440.wav')
#print(len(data1))
#data2 = writeWave('TestFiltered.wav', data1, rate)
def trainModel(lib):
    #sess = tf.InteractiveSession()
    library = scanLibrary(lib)
    book = library[0]
    origin1, origin2, sizeo = readWave(book[0], 10)
    dest1, dest2, sized = readWave(book[1], 10)
    if sizeo != sized:
        print("Error sizeo != sized")
        return
    
    #model = keras.Sequential([
        #keras.layers.Flatten(input_shape=(44100, 2)),
    #    keras.layers.Flatten(),
    #    keras.layers.Dense(sizeo*2, activation=tf.nn.relu),
    #    keras.layers.Dropout(0.2),
    #    keras.layers.Dense(sized*2, activation=tf.nn.softmax)
    #])

    #model.compile(optimizer=tf.train.AdamOptimizer(),
    #        loss='sparse_categorical_crossentropy',
    #        metrics=['accuracy'])
    #for i in range(len(origin)):
    #    origin[i]=tf.convert_to_tensor(origin[i])
    #origintens = tf.convert_to_tensor(origin)
    #for i in range(len(dest)):
    #    dest[i]=dest[i].flatten()

    #desttens = tf.convert_to_tensor(dest)
    #print(desttens)
    origin1=np.array(origin1)

    dest1=np.array(dest1)
    print(origin1.shape)
    print(dest1.shape)

    #model.fit(origin, dest)

    #return model, library, origin, dest
    #return model
    return None

#for book in library:
if __name__=='__main__':
    #trainModel(sys.argv[1])
    trainModel('trainer/samples-44100')
