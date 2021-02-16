import numpy as np 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data.dataset import random_split
#import dataset
import time
import logging 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dtype', metavar='data type',
                    help='image or audio')
parser.add_argument('--label', metavar='data type',
                    help='label name ex) gender, emotion, gaze, tone, intent') 
parser.add_argument('--num_classes', metavar='data type', type=int,
                    help='number of label classes ex) gender 2 emotion gaze tone 5 intent 6')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')  
parser.add_argument('--wd', default=0, type=float, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--gpu', default=3, type=float, metavar='N',
                    help='gpu id') 
parser.add_argument('--start', default=0, type=int, metavar='N',
                    help='start epoch') 
parser.add_argument('--resume', default=False, type=bool, metavar='N',
                    help='resume train') 
args = parser.parse_args()  




# Device configuration
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'gpu device {device}')
 


logging.basicConfig(format='%(message)s',filename='train.log', filemode='w',level=logging.INFO)
logger = logging.getLogger()
 


start = time.time()
# Data loading code
#data_set = dataset.TrainingDataset(data_dir=args.data, datype=args.dtype, label=args.label)
 
#train_dataset, valid_dataset, test_dataset = random_split(data_set, [int(round(len(data_set)*0.75)),int(round(len(data_set)*0.05)), int(round(len(data_set)*0.2))])

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.load('./train_data.npy')),torch.from_numpy(np.load('./train_label.npy')))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.load('./valid_data.npy')),torch.from_numpy(np.load('./valid_label.npy')))



train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers)

#valid_loader = torch.utils.data.DataLoader(
#    valid_dataset, batch_size=args.batch_size, shuffle=True,
#    num_workers=args.workers)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4))  

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4))
        self.fc1 = nn.Linear(8960,4000)
        self.fc2 = nn.Linear(4000,1000)
        self.fc3 = nn.Linear(1000,num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out) 
        out = self.fc3(out)
        return out

model = ConvNet(args.num_classes).to(device)

if args.resume:
    model.load_state_dict(torch.load('./intent_model.pt',map_location=device))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)

train_time = time.time()-start

# logger.info(f'time taken for loading dataset : {train_time}')

def validation(model, data_loader):

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0 
        global device
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.squeeze()   
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()     
        test_acc = correct / total 
    return test_acc, correct, total


best_acc = 0.0 

# Train the model 
for epoch in range(args.start, args.start+args.num_epochs):
    running_loss = 0.0
    running_corrects = 0
    model.train() 

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device) 
        labels = labels.squeeze()   
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) 
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        _, predicted = torch.max(outputs.data, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(predicted == labels)
 
    epoch_loss = running_loss / len(train_dataset) 
    epoch_acc = running_corrects.double() / len(train_dataset) 


    #val_acc, _ , _ = validation(model,valid_loader)

    #if val_acc > best_acc:
       # best_model = model.copy()
 
    logger.info('Epoch [{}/{}] Loss: {:.4f} Train Acc: {:.4f}'.format(
        epoch+1, args.num_epochs, epoch_loss, epoch_acc))

    test_acc, correct, length = validation(model,test_loader)

    print('Epoch {} Test Accuracy of the model : [{}/{}]  {:.3f}'.format(
        epoch+1, correct, length, 100* test_acc))
    if test_acc > 0.7:
        logger.info('Epoch {} Test Accuracy of the model : [{}/{}]  {:.3f}'.format(
        epoch+1, correct, length, 100* test_acc))
        torch.save(model.state_dict(), './weight/{}_{:.2f}.pt'.format(epoch, test_acc))
# print(f'time taken for loading dataset : {train_time}') 
logger.info(f'time taken for training {time.time()-train_time}')
# Test the model  

test_acc, correct, length = validation(model,test_loader)

logging.info('Test Accuracy of the model : [{}/{}]  {:.3f}'.format(
    correct, length, 100 * test_acc))
# logging.info(f'correct {correct} total {length}')

#acc , correct, length = validation(best_model,test_loader)
#logging.info('Test Accuracy of the best model : [{}/{}]  {:.3f}'.format(
#    correct, length, 100* acc))

# if test_acc >= 0.7 or acc >= 0.7:
torch.save(train_dataset, 'train_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
# Save the model checkpoint
torch.save(model.state_dict(), 'intent_model.pt') 