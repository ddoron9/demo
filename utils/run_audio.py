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

print(os.path.abspath('.'))  

from models.audio_emotion import *
from models.audio_gender import *
from utils.audio_utils import *

global emotion_classes 
emotion_classes = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'} 
global gender_classes
gender_classes = ['Female','Male']

global SAMPLING_RATE 
SAMPLING_RATE = 48000  
end = 0
ringBuffer = RingBuffer(48000 * 3)
ringBuffer.extend(np.zeros( 48000 * 3))
pa = None
stream = None

def predict(model,X):
    with torch.no_grad():
        model.eval()
        outputs = model(X) 
    return outputs

def callback(in_data, frame_count, time_info, flag):  
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    ringBuffer.extend(audio_data)
    return None, pyaudio.paContinue

def audio_emotion_class(video):
    '''
    audio path and ringBuffer end
    '''
    if(not ringBuffer.is_full):
        return

    mel_spectogram = getMELspectrogram(np.array(ringBuffer).astype('float32'), SAMPLING_RATE)
    
    mel_spectogram-=mel_spectogram.min()
    mel_spectogram/=mel_spectogram.max()

    im = np.uint8(cm.gist_earth(mel_spectogram)*255)[:,:,:3]
    imagesTensor = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
        transforms.Grayscale(num_output_channels=1),
    ])(im).view(1,1,im.shape[0],im.shape[1])

    gender_outputs = predict(gender_model,imagesTensor.to(device))
    emotion_outputs = predict(emotion_model,imagesTensor.to(device))
    
    prediction = torch.argmax(emotion_outputs[1],dim=1)
    gender = torch.argmax(gender_outputs,dim=1)

    if video:
        global end
        ringBuffer.extend(audio[end:end+int(SAMPLING_RATE/20)])
        end += int(SAMPLING_RATE/20)
    return gender_classes[int(gender)], emotion_classes[int(prediction)]

def audio_inference(video):
    if video is None:
        stream.start_stream()  
    return audio_emotion_class(video)
        
def stop_audio():
    global pa 
    time.sleep(1)
    stream.close()

def start_audio(ModelPath, video : str): 
    print("Loading all relevant data.")
    
    global device
    global emotion_model
    emotion_model = AucousticEmotion(len(emotion_classes))
    device = torch.device('cpu')
    emotion_model.load_state_dict(torch.load(ModelPath+'cnn_transf_parallel_model.pt', map_location=device)) 
    
    global gender_model
    gender_model = AucousticGender()
    gender_model.load_state_dict(torch.load(ModelPath+'audio_gender.pt', map_location=device)) 
    
    if video:
        '''
        audio file load
        '''
        global audio
        audio, sample_rate = librosa.load(video, sr=SAMPLING_RATE)
        ringBuffer.extend(audio[:end])
    else:
        '''
        stream mode
        '''
        global stream
        print("Opening Audio Channel")
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLING_RATE, 
                        input=True,
                        frames_per_buffer=int(48000/20),
                        stream_callback=callback) 
        stream.start_stream()

    print("Starting Running")