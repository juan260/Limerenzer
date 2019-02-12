import tensorflow as tf
import numpy as np
from libLoader import *
import random
import dill
import os
import shutil

class BrainError (Exception):
    def __init__(self, value='Error in the brain'):
        self.value=value
    def __str__(self):
        return self.value
class LimeBrain:
    def __init__(self, input_dim, hidden_dim, output_dim, epoch=250, learning_rate=0.01, name='Give_me_a_name'):
        self.epoch=epoch
        self.learning_rate = learning_rate
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.name=name

        self.buildBrain()

        path='./data/'+name
        try:
            os.mkdir('./data')
        except OSError:
            pass
        try:
            os.mkdir(path)
        except OSError:
            shutil.rmtree(path)
            os.mkdir(path)

    def buildBrain(self):
        input_data=tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        output_data=tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim])
        with tf.name_scope('encode'):
            #weights=tf.Variable(tf.complex(
            #        tf.random_normal([input_dim, hidden_dim]),
            #        tf.random_normal([input_dim, hidden_dim])),
            #    dtype=tf.float32, name='weights')
            weights=tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim]), dtype=tf.float32, name='weights')
            biases=tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32), name='biases')
            encoded = tf.nn.tanh(tf.matmul(input_data, weights) + biases)

        with tf.name_scope('decode'):
            #weights=tf.Variable(tf.complex(
            #        tf.random_normal([input_dim, hidden_dim]),
            #        tf.random_normal([input_dim, hidden_dim])),
            #    dtype=tf.float32, name='weights')
            weights=tf.Variable(tf.random_normal([self.hidden_dim, self.output_dim]), dtype=tf.float32, name='weights')
            biases=tf.Variable(tf.zeros([self.output_dim], dtype=tf.float32), name='biases')
            decoded = tf.matmul(encoded,weights)+biases

        self.input_data=input_data
        self.output_data=output_data
        self.encoded=encoded
        self.decoded=decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output_data, self.decoded))))
        self.train_op=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        #self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver()

    def train(self, in_train_data, out_train_data):
        for i in range(len(in_train_data)):
            if self.input_dim != len(in_train_data[i]):
                raise BrainError('The input data has the wrong size. Expecting \
{0} but recieved {1} on iteration {2}'.format(int(self.input_dim), len(in_train_data[i]), i))
        for i in range(len(out_train_data)):
            if self.output_dim != len(out_train_data[i]):
                raise BrainError('The output data has the wrong size. Expecting \
{0} but recieved {1}'.format(int(self.output_dim), len(out_train_data[i])))

        if len(in_train_data)!=len(out_train_data):
            raise BrainError('The input training data has a different length than the out training data')

        num_samples=len(in_train_data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for j in range(num_samples):
                    l, _ = sess.run([self.loss, self.train_op],
                        feed_dict={self.input_data: [in_train_data[j]],
                        self.output_data: [out_train_data[j]]})

                if i%10:
                    print('epoch {0}: loss = {1}'.format(i,l))
                    self.saver.save(sess, './data/'+self.name+'/model.ckpt')
            self.saver.save(sess, './data/'+self.name+'/model.ckpt')
        self.trained=True


    def run(self, dataIn):
        if not self.trained:
            raise BrainError('The model has not been trained yet!!')
        if len(dataIn[0]) != self.input_dim:
            raise BrainError('The input data doesn\'t match the input neuron size \
Expected {0} and got {1}'.format(len(dataIn), self.input_dim))

        with tf.Session() as sess:
            self.saver.restore(sess, './data/'+self.name+'/model.ckpt')
            result=  sess.run([self.decoded],
                feed_dict={self.input_data : dataIn})
        return result[0]

    def dumpBrain(self, *name):
        input_data=self.input_data
        output_data=self.output_data
        saver=self.saver
        encoded=self.encoded
        decoded=self.decoded
        loss=self.loss
        train_op=self.train_op
        self.input_data=None
        self.output_data=None
        self.saver=None
        self.encoded=None
        self.decoded=None
        self.loss=None
        self.train_op=None
        if len(name)==0:
            directory=self.name
        else:
            directory=name[0]
        #if not os.path.isfile('./data/'+directory+'/brain.obj'):

        with open('./data/'+directory+'/brain.obj', 'w') as f:
            dill.dump(self, f)
        self.input_data=input_data
        self.output_data=output_data
        self.saver=saver
        self.encoded=encoded
        self.decoded=decoded
        self.loss=loss
        self.train_op=train_op

def loadBrain(name):
    with open('./data/'+name+'/brain.obj') as f:
        loaded= dill.load(f)
    loaded.buildBrain()
    return loaded

if __name__ == '__main__':
    lib= 'trainer/samples-44100'
    #library = scanLibrary(lib)
    #book = library[0]
    division = 20
    #origin, sizeo = readWave(book[0], division)
    #dest, sized = readWave(book[1], division)

    # #for i in range(len(origin)):
    # #    origin[i]=origin[i].flatten()

    # #for i in range(len(dest)):
    # #    dest[i]=dest[i].flatten()

    # #TODO: descomentar estas lineas:
    # #oridest = zip(origin,dest)
    # #random.shuffle(oridest)
    # #origin,dest=zip(*oridest)
    parts=100
    origin, dest, sizeo, sized = loadLibrary(lib, division, parts, 0)
    print('Loaded library')
    brains = LimeBrain(sizeo, sizeo, sized, epoch = 5, learning_rate=0.1, name='frankenstein v. 2')
    print('Created frankenstein v. 2 with sizeo {0} and sized {1}'.format(sizeo, sized))
    print('Starting first training')

    brains.train(origin, dest)
    for i in range(1, parts):
        print('Starting training {0}'.format(i))
        origin, dest, sizeo, sized = loadLibrary(lib, division, parts, i)
        print('Loaded library')
        brains.train(origin, dest)
        print('Trining complete')
    brains.dumpBrain()
    #brains=loadBrain('frankenstein')
    print('Dump complete')
    data, size = readWave('./trainer/samples-44100/1/Piano/-3-100-d.wav', division)
    print('Running model')
    result = brains.run(data)
    writeWave('PruebaFinal.wav',result , size, division)



