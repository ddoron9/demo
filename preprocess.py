import os
import glob
import json
import numpy as np
import librosa
from utils.audio_utils import getMELspectrogram, addAWGN
from matplotlib import cm
from torchvision import transforms, models
import torch
from tqdm import tqdm
from torch.utils.data import  TensorDataset, DataLoader


    

class AudioData:
    def __init__(self,path,SAMPLE_RATE):
        self.sr = SAMPLE_RATE
        self.emotion_classes = {'중립' : 0, '슬픔' : 1, '놀람' : 2, '행복' : 3, '화남' : 4} 
        self.gender_classes = {'남자' : 0, '여자' : 1}
        self.intention_classes = {'요청-위치안내' : 0, '요청-기타안내' : 1, '질문-위치' : 2, '질문-기타' : 3, '인사' : 4, '진술' : 5}
        self.gender = None
        self.tone = None
        self.intent = None
        self.data = np.zeros((1,3))
        self.path = path

    def json_reorder(self,curdata):
        gender_index = -1
        intent_index = -1
        tone_index = -1
        for i, c in enumerate(curdata):
            if c["from_name"] == 'bbox':
                gender_index = i
            if c["from_name"] == 'intent':
                intent_index = i
            if c["from_name"] == 'tone':
                tone_index = i
        return gender_index, intent_index, tone_index

    def audio2mels(self,curwav):
        audio, sample_rate = librosa.load(curwav, duration=3, offset=0.5, sr=self.sr)
        signal = np.zeros((int(self.sr*3,)))
        signal[:len(audio)] = audio

        augmented_signal = addAWGN(signal)
        augmented_signal = augmented_signal.reshape(-1) 
        try:
            curdata = [[self.gender_classes[self.gender], self.intention_classes[self.intent], self.emotion_classes[self.tone]]]
            self.data = np.append(self.data,curdata,axis = 0)
        except Exception as e:
            print(e)
            return 
        mel_spectrogram = getMELspectrogram(augmented_signal, sample_rate=self.sr) 
        mel_spectrogram-=mel_spectrogram.min()
        mel_spectrogram/=mel_spectrogram.max()

        # im = np.uint8(cm.gist_earth(mel_spectrogram)*255)[:,:,:3]
        imagesTensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5),
            # transforms.Grayscale(num_output_channels=1),
        ])(mel_spectrogram[np.newaxis,:])
        # ])(im)
        return imagesTensor.permute(1,2,0)

    def get_audio(self):
        p = os.listdir(self.path)
        directories = []
        for i in p:
            if os.path.isdir(self.path + '/' + i):    
                directories.append(self.path+'/'+i)

        num_labeled = 0
        num_unlabeled = 0
 
        for d in tqdm(directories):
            waves = glob.glob(d + '/*.wav')
            jsons = glob.glob(d + '/*.json')

            if len(waves)== 0 or len(jsons) == 0:
                num_unlabeled +=1
                continue
            else:
                curwav = waves[0]
                curjsons = jsons
                gender = ""
                intent = ""
                tone = ""
                for curjson in curjsons:
                    with open(curjson) as f:
                        curdata = json.load(f)
                    gender_index, intent_index, tone_index = self.json_reorder(curdata)
                    try:
                        self.gender = curdata[gender_index]["value"]["rectanglelabels"][0]#성별
                    except Exception as e:
                        pass
                    try:
                        self.intent = curdata[intent_index]["value"]["choices"][0]#의도
                    except Exception as e:
                        pass
                    try:
                        self.tone = curdata[tone_index]["value"]["choices"][0]#어조
                    except Exception as e:
                        pass
                    f.close()
                num_labeled += 1
                imagesTensor = self.audio2mels(curwav)
                if imagesTensor == None:
                    continue
                try :  
                    stacked_mel = torch.cat([stacked_mel, imagesTensor]) 
                except NameError as e:
                    stacked_mel = imagesTensor

        stacked_mel = stacked_mel.unsqueeze(1)
        self.data = torch.from_numpy(self.data[1:,:])
        
        return stacked_mel, self.data

# PATH = '/opt/utopsoft/label-management/webapps/attach/label'
# a = AudioData(PATH,48000)
# data, label = a.get_audio()
# print(data.size(),label.size())
# torch.save(data, './train_data.pt')
# torch.save(label, './train_label.pt')
