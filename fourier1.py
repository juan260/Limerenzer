import scipy
import wave
import struct
import numpy as np


from scipy.io import wavfile

rate, data = wavfile.read('More complex audio.wav')

filtereddata = np.fft.fft(data, axis=0)

filteredComplexWrite = np.fft.ifft(filtereddata,axis=0)

filteredComplexWrite[0]=np.float64(-1)
filteredRealWrite=[np.int16(-1)]*len(filteredComplexWrite) 
for i in range(1, len(filteredComplexWrite)):
	filteredRealWrite[i]=np.int16(np.rint(\
		np.real(np.sign(filteredComplexWrite[i]))*\
		np.abs(filteredComplexWrite[i])))

filteredRealWrite=np.array(filteredRealWrite)

#print(len(filteredRealWrite))
#print(len(data))
if(len(data)!=len(filteredRealWrite)):
	print("CUIDAITO")
for i in range(len(filteredRealWrite)):
	if(i%1000==0):
		print("Llevamos " + str(i) + " de " + str(len(filteredRealWrite)))
	if(data[i]!=filteredRealWrite[i]):
		print("ERROR: " +str(data[i]) + "\t\t" + str(filteredRealWrite[i]) + "\t\t" + str(filteredComplexWrite[i]))

print("ESCRIBIENDO")
wavfile.write('TestFiltered.wav', rate, filteredRealWrite)
