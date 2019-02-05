from fourier1 import *
w1, s = readWave('trainer/samples-44100/0/Piano/0-100-d.wav', 10)
#w2, s = readWave('avekeapasao.wav', 10)

writeWave('avekeapasao.wav', w1, s, 10)
rate1, dat1 = wavfile.read('trainer/samples-44100/0/Piano/0-100-d.wav')
rate2, dat2 = wavfile.read('avekeapasao.wav')
counter =0
for i in range(len(dat1)):
    if(dat1[i][0]!=dat2[i][0] or dat1[i][1]!=dat2[i][1]):
        counter+=1
        print('I: ' + str(i) + ' Count: ' + str(counter) + ' DAT 1: ' + str(dat1[i]) + ' DAT 2: ' + str(dat2[i]))
