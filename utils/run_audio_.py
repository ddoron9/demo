import queue
import matplotlib.pyplot as plt
from threading import Thread
import pyaudio
import librosa
import numpy as np
from matplotlib import cm
import pickle
import os, sys

import torch
import torch.nn as nn
from torchvision import transforms, models
import time
from numpy_ringbuffer import RingBuffer
from models.audio_emotion import *
from models.audio_gender import *
from utils.audio_utils import *


class AudioDetector():
    
    def __init__(self):
        self.emotion_classes = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'} 
        self.gender_classes = ['Female','Male']
        self.sr = 48000  
        self.end = 0
        self.ringBuffer = RingBuffer(48000 * 3)
        self.ringBuffer.extend(np.zeros( 48000 * 3))
        self.pa = None
        self.stream = None
        self.gender_model = None
        self.emotion_model = None
        self.device = None
        self.audio = None 
        
    def predict(model,X):
        with torch.no_grad():
            model.eval()
            outputs = model(X) 
        return outputs

    def callback(self,in_data, frame_count, time_info, flag):  
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.ringBuffer.extend(audio_data)
        return None, pyaudio.paContinue

    def audio_emotion_class(self,video):
        '''
        audio path and ringBuffer end
        '''
        if(not self.ringBuffer.is_full):
            return

        mel_spectogram = getMELspectrogram(np.array(self.ringBuffer).astype('float32'), self.sr)
        
        mel_spectogram-=mel_spectogram.min()
        mel_spectogram/=mel_spectogram.max()

        im = np.uint8(cm.gist_earth(mel_spectogram)*255)[:,:,:3]
        imagesTensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5),
            transforms.Grayscale(num_output_channels=1),
        ])(im).view(1,1,im.shape[0],im.shape[1])

        gender_outputs = self.predict(gender_model,imagesTensor.to(self.device))
        emotion_outputs = self.predict(emotion_model,imagesTensor.to(self.device))
        
        prediction = torch.argmax(emotion_outputs[1],dim=1)
        gender = torch.argmax(gender_outputs,dim=1)

        if video: 
            self.ringBuffer.extend(self.audio[self.end:self.end+int(self.sr/20)])
            self.end += int(self.sr/20)
        return self.gender_classes[int(gender)], self.emotion_classes[int(prediction)]

    def audio_inference(self,video):
        if video is None:
            self.stream.start_stream()  
        return audio_emotion_class(video)
            
    def stop_audio(self): 
        time.sleep(1)
        self.stream.close()

    def start_audio(self,ModelPath, video : str): 
        print("Loading all relevant data.")
         
        self.emotion_model = AucousticEmotion(len(self.emotion_classes))
        self.device = torch.device('cpu')
        self.emotion_model.load_state_dict(torch.load(ModelPath+'cnn_transf_parallel_model.pt', map_location=self.device)) 
        
        self.gender_model = AucousticGender()
        self.gender_model.load_state_dict(torch.load(ModelPath+'audio_gender.pt', map_location=self.device)) 
        
        if video:
            '''
            audio file load
            ''' 
            self.audio, sample_rate = librosa.load(video, sr=self.sr)
            self.ringBuffer.extend(self.audio[:end])
        else:
            '''
            stream mode
            ''' 
            print("Opening Audio Channel")
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sr, 
                            input=True,
                            frames_per_buffer=int(48000/20),
                            stream_callback=self.callback) 
            self.stream.start_stream()

        print("Starting Running")