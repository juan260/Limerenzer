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

def unComplex(fft):
    #l = [None for _ in range(len(fft)*2)]
    #i = 0
    #for compl in fft:
    #    l[i] = np.abs(compl)
    #    l[i+1] = np.angle(compl)
    #    i+=2
    for i in range(len(fft)):
        fft[i]= np.append(np.abs(fft[i]), np.angle(fft[i]))
    return fft

def reComplex(fft):
    #l = [None for _ in range(len(fft)/2)]
    #i=0
    #while i< range(len(fft)):
    #    l[i//2] = radii * exp(1j*angles)
    for i in range(len(fft)):
        radii=fft[i][:len(fft[i])//2]
        angles=fft[i][len(fft[i])//2:]
        comp = 1j*angles
        exp=np.exp(comp)
        fft[i]=radii * exp
    return fft


def fastFourier(data):
    return np.fft.fft(data, axis=0)

def inverseFastFourier(filtereddata):
    
    filteredComplexWrite = np.fft.ifft(filtereddata,axis=0)

    #filteredComplexWrite[0]=np.float64(-1)
    filteredRealWrite=[np.int16(-1)]*len(filteredComplexWrite)
    for i in range(1, len(filteredComplexWrite)):
        filteredRealWrite[i]=np.int16(np.rint(\
	    np.real(np.sign(filteredComplexWrite[i]))*\
	    np.abs(filteredComplexWrite[i])))
    
    filteredRealWrite=np.array(filteredRealWrite)
    #toWrite=filteredRealWrite[0]
    #for i in range(1, len(filteredRealWrite)):
    #    toWrite.append(filteredRealWrite[i])

    #return toWrite
    return filteredRealWrite

def chunkAndFFT(data, size):
    chunks = [data[x:x+size] for x in xrange(0, len(data), size)]
    for i in range(len(chunks)):
        chunks[i] = fastFourier(chunks[i])
    return chunks

def readWave(path, division, samples=-1):

    rate, data = wavfile.read(path)
    if rate%division:
        print('No se puede dividir el rate ' + str(rate) +\
            ' entre ' + str(division))
        return None
    size = rate/division
    if samples>=0:
        data=data[:samples]
    # Divide data into two separate channels and return
    channel1 = [el[0] for el in data]
    channel2 = [el[1] for el in data]
    # Divide into chunks

    fft1=chunkAndFFT(channel1, size)
    #print len(fft1)
    #print len(fft1[1])
    fft2=chunkAndFFT(channel2, size)

    # Divide the complex values into real
    real1=unComplex(fft1)
    real2=unComplex(fft2)
    #return np.array(real1), np.array(real2), size*2
    return appendChannels(np.array(real1), np.array(real2)), size*4

def appendChannels(chan1, chan2):
    l=[None]*len(chan1)
    #l = np.append(chan1, chan2)
    for i in range(len(l)):
        l[i]=np.append(chan1[i], chan2[i])
        l[i][::2]=chan1[i]
        l[i][1::2]=chan2[i]
    return l
    #return chan1.append(chan2)

def separateChannels(data):
    chan1=[None]*len(data)
    chan2=[None]*len(data)
    for i in range(len(data)):
        chan1[i]=data[i][::2]
        chan2[i]=data[i][1::2]
    return chan1, chan2
    #return data[:len(data)//2], data[len(data)//2:]

def soft(data, size):
    i=1
    for i in range(size, len(data), size//2):
        data[i] = np.int16(np.rint((data[i-1]+data[i+1])/2))
    #data[0] = [30, 30]


    return data

def writeWave(path, filtereddata, size, division):
    filtereddata1, filtereddata2=separateChannels(filtereddata)
    filtereddata1=reComplex(filtereddata1)
    filtereddata2=reComplex(filtereddata2)
    
    for i in range(len(filtereddata1)):
        filtereddata1[i]=inverseFastFourier(filtereddata1[i])
    for i in range(len(filtereddata2)):
        filtereddata2[i]=inverseFastFourier(filtereddata2[i])

    unchunked1=list(itertools.chain.from_iterable(filtereddata1))
    unchunked2=list(itertools.chain.from_iterable(filtereddata2))
    unchunked1=soft(unchunked1, size)
    unchunked2=soft(unchunked2, size)
    toWrite=np.array(zip(unchunked1, unchunked2))
    wavfile.write(path, division*size//4, toWrite)
    return toWrite




#data1, rate = readWave('440.wav')
#print(len(data1))
#data2 = writeWave('TestFiltered.wav', data1, rate)
# def trainModel(lib):
#     #sess = tf.InteractiveSession()
#     library = scanLibrary(lib)
#     book = library[0]
#     origin1, origin2, sizeo = readWave(book[0], 10)
#     dest1, dest2, sized = readWave(book[1], 10)
#     if sizeo != sized:
#         print("Error sizeo != sized")
#         return

#     #model = keras.Sequential([
#         #keras.layers.Flatten(input_shape=(44100, 2)),
#     #    keras.layers.Flatten(),
#     #    keras.layers.Dense(sizeo*2, activation=tf.nn.relu),
#     #    keras.layers.Dropout(0.2),
#     #    keras.layers.Dense(sized*2, activation=tf.nn.softmax)
#     #])

#     #model.compile(optimizer=tf.train.AdamOptimizer(),
#     #        loss='sparse_categorical_crossentropy',
#     #        metrics=['accuracy'])
#     #for i in range(len(origin)):
#     #    origin[i]=tf.convert_to_tensor(origin[i])
#     #origintens = tf.convert_to_tensor(origin)
#     #for i in range(len(dest)):
#     #    dest[i]=dest[i].flatten()

#     #desttens = tf.convert_to_tensor(dest)
#     #print(desttens)
#     origin1=np.array(origin1)

#     dest1=np.array(dest1)
#     print(origin1.shape)
#     print(dest1.shape)

#     #model.fit(origin, dest)

#     #return model, library, origin, dest
#     #return model
#     return None

#for book in library:
# if __name__=='__main__':
#     #trainModel(sys.argv[1])
#     trainModel('trainer/samples-44100')
