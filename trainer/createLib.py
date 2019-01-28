import mingus.core.notes
from mingus.containers import *
from mingus.containers.instrument import *
from mingus.midi import fluidsynth
from mingus.midi.fluidsynth import FluidSynthSequencer
from mingus.midi.sequencer import Sequencer
from mingus.containers.instrument import MidiInstrument
import mingus.midi.pyfluidsynth as fs
import time
import wave
import os
import shutil

SOUND_FONT=os.path.join(os.getcwd(),'Nice-Keys-Suite-V1.0.sf2')
#fluidsynth.init(SOUND_FONT, 'alsa')

MAX_NOTE=Note('C', 8)

class HarmonyModel():
    modelStart=[]
    modelEnd=[]
    minimum=0
    maximum=int(MAX_NOTE)

    def __init__(self, minimum=Note('C', 0), maximum=int(MAX_NOTE),modelStart=[], modelEnd=[]):
        self.minimum=int(minimum)
        self.maximum=int(maximum)
        self.modelStart=modelStart
        self.modelEnd=modelEnd

    def dump(self, seconds, wav, synth, mili=False):
        if mili:
            seconds=seconds/1000
        samples = fs.raw_audio_string(synth.get_samples(\
                int(seconds * 44100)))
        wav.writeframes(''.join(samples))

    def dumpNotes(self, path, synth, notes, tiempo, velocity, aumentation):

        w = wave.open(path, 'wb')
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        #self.dump(100, w, synth, mili=True)
        for note in notes:
            #TODO: aniadir velocidades distintas
            synth.noteon(1, int(note)+12+aumentation, velocity)

        self.dump(tiempo, w, synth)

        for note in notes:
            synth.noteoff(1, int(note)+12+aumentation)
        w.close()

    def generalize(self, instruments):
        # First we get the lowest and the highest notes in the models
        minimalNote=int(MAX_NOTE)
        maximalNote=0

        for note in self.modelStart + self.modelEnd:
            if int(note) < minimalNote:
                minimalNote=int(note)
            if int(note) > maximalNote:
                maximalNote=int(note)

        #startBar = Bar()
        #endBar = Bar()
        #startBar.place_notes(self.modelStart, 1)
        #endBar.place_notes(self.modelEnd, 1)

       #sequencer = FluidSynthSequencer()
        synth = fs.Synth()
        synth.sfload(SOUND_FONT)
        #synth.start('alsa')
        synth.program_reset()
        #print(sequencer.init(SOUND_FONT,'alsa'))

        aumentations = range(self.minimum-minimalNote, self.maximum-maximalNote)
        for instrument in instruments:
            i=0
            synth.program_change(1, i)
            synth.program_reset()
            #startTrack=Track(instrument)
            #endTrack=Track(instrument)
            #startTrack.add_bar(startBar)
            #endTrack.add_bar(endBar)
            #sequencer.start_recording('helo.wav')
            #sequencer.fs.program_reset()
            #sequencer.play_Track(startTrack)
            os.mkdir(str(instrument))
            os.chdir(str(instrument))
            for aumentation in aumentations:
                velocity=100
                time=10
                name=str(aumentation)+'-'+str(velocity)
                self.dumpNotes(name+'-o.wav', synth, self.modelStart, time, velocity, aumentation)

                self.dumpNotes(name+'-d.wav', synth, self.modelEnd, time, velocity, aumentation)
            os.chdir('../')
            i+=5
            #sequencer.stop_everything()
            #sequencer.play_Track(endTrack)
            #fluidsynth.play_Track(endTrack)
        #aumentation=self.minimum-minimalNote

        #while aumentation+maximalNote < self.maximum:


        #    aumentation+=1
        synth.delete()

def exportSamples(harmonyModels, instruments):
    #TODO: Aniadir rate
    name='samples-44100'
    try:
        os.mkdir(name)
    except OSError:
        shutil.rmtree(name)
        os.mkdir(name)
    os.chdir(name)
    i=0
    for model in harmonyModels:
        os.mkdir(str(i))
        os.chdir(str(i))
        model.generalize(instruments)
        os.chdir('../')
        i+=1
    os.chdir('../')

h = [HarmonyModel(modelStart=[Note('C', 4)], modelEnd=[Note('C', 4), Note('E', 4), \
        Note('G', 4)],minimum=Note('C', 2), maximum=Note('G', 6)),

    HarmonyModel(modelStart=[Note('C', 4), Note('E', 4), Note('G', 4)],
        modelEnd=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B', 4)],
        minimum=Note('E', 2), maximum=Note('G', 6)),

    HarmonyModel(modelStart=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B', 4)],
        modelEnd=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B', 4), Note('D', 5)],
        minimum=Note('E', 2), maximum=Note('G', 6)),

    HarmonyModel(modelStart=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B',4)],
        modelEnd=[Note('C', 4), Note('E', 4), Note('B', 4), Note('A', 4), Note('D', 5)],
        minimum=Note('B', 2), maximum=Note('G', 6)),

    HarmonyModel(modelStart=[Note('C', 4), Note('E', 4), Note('B', 4), Note('A', 4), Note('D', 5)],
        modelEnd=[Note('C', 4), Note('E', 4), Note('B', 4), Note('A', 4), Note('D', 5), Note('D', 4),Note('E', 5)],
        minimum=Note('B', 2), maximum=Note('G', 6)),

    HarmonyModel(modelStart=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B', 4), Note('D', 5)],
        modelEnd=[Note('C', 4), Note('E', 4), Note('G', 4), Note('B', 4), Note('D', 4), Note('D', 5), Note('A', 4)],
        minimum=Note('B', 2), maximum=Note('G', 6))]


#next(os.walk('.'))[1]
#h.generalize([Piano()])
exportSamples(h, ['Piano','OtroPiano'])
