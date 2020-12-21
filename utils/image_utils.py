import cv2
import numpy as np
import torch
from torchvision import transforms
from models import mini_xception, resnet34

def draw_bbox(frame, boxes, probs):
    '''
    Draw bounding box and probs
    '''
    for box, prob in zip(boxes, probs):
        # draw rectangle on frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,0,255), thickness=2)

        # show probability
        cv2.putText(frame, str(prob), (box[2], box[3]),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return frame

def detect_rois(boxes):
    '''
    return rois as a list
    '''
    rois = list()
    for box in boxes:
        roi = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        rois.append(roi)
    return rois

def gender_class(face):
    '''
    gender classification
    '''
    # convert color scale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # resize & make tensor
    face = cv2.resize(face, (48,48))
    face = np.array(face)
    img_tensor = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize([0.5,],[0.5,])])(face)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    #print(img_tensor.shape)

    # gender predict
    labels = ['woman', 'man']
    device = torch.device('cpu')
    model = mini_xception.Model(num_classes=2)
    weight = './trained_model/gender_xception_0.08.pt'
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    with torch.no_grad():
        data = img_tensor.to(device)
        out = model(data)
    return labels[out.argmax(dim=1, keepdim=True)]

def emotion_class(face):
    '''
    emotion recognition from image
    '''
    # convert color scale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # resize & make tensor
    face = cv2.resize(face, (48,48))
    face = np.array(face)
    img_tensor = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize([0.5,],[0.5,])])(face)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    #print(img_tensor.shape)

    # emotion predict
    labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    device = torch.device('cpu')
    model = mini_xception.Model(num_classes=5)
    weight = './trained_model/emotion_xception_1.34.pt'
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    with torch.no_grad():
        data = img_tensor.to(device)
        out = model(data)
    return labels[out.argmax(dim=1, keepdim=True)]

def gaze_class(face):
    '''
    gaze recognition
    '''
    # convert color scale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # resize & make tensor
    face = cv2.resize(face, (48,48))
    face = np.array(face)
    img_tensor = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize([0.5,],[0.5,])])(face)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    #print(img_tensor.shape)

    # gaze predict
    labels = ['center', 'down', 'left', 'right', 'up']
    device = torch.device('cpu')
    model = resnet34.Model(False)
    weight = './trained_model/headpose_resnet34.pth'
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    with torch.no_grad():
        data = img_tensor.to(device)
        out = model(data)
    return labels[out.argmax(dim=1, keepdim=True)]
