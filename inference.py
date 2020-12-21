import torch
import torchvision.transforms as transforms
from models import senet, mini_xception, custom
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn as nn
import torchvision.models as models

mtcnn = MTCNN()
data = 'data\\image\\image.jpg'
cap = cv2.VideoCapture(data)
ret, frame = cap.read()
boxes, probs = mtcnn.detect(frame, landmarks=False)

def detect_rois(boxes):
    rois = list()
    for box in boxes:
        roi = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        rois.append(roi)
    return rois

rois = detect_rois(boxes)

for roi in rois:
    (start_Y, end_Y, start_X, end_X) = roi
    face = frame[start_Y:end_Y, start_X:end_X]

color = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
f = cv2.resize(color, (224,224))
f = np.array(f)
print(f.shape)
trans_tensor = transforms.ToTensor()
f = trans_tensor(f)
print(f.shape)
trans_norm = transforms.Normalize([0.5,],[0.5,])
f = trans_norm(f)

f = torch.unsqueeze(f, 0)
print(f.shape)

model = models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

model.load_state_dict(torch.load('trained_model\\headpose_resnet34_state-dict_epoch20_2.pth', map_location='cpu'))

model.eval()

output = model(f)
print(output)
preds = output.argmax(dim=1, keepdim=True)

GAZE = ['center', 'down', 'left', 'right', 'up']
label = GAZE[preds]
print(label)
