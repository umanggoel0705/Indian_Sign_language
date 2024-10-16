import torch
import torchvision.transforms as T
from torch import nn
import numpy as np
import pandas as pd
import cv2
from PIL import Image

device = "cuda"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(16*16*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 35),
            nn.Softmax()
        )
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model_99%.pth'))
model.eval()
# print(model)

transform = T.Compose([T.Resize((128,128)), T.ToTensor()])

target_names = ['1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','I',
                'J','K','L','M','N','O','P','Q','R',
                'S','T','U','V','W','X','Y','Z']

prev_char = ''
ouput_string = ""
cap = cv2.VideoCapture('video1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (420, 420))

        img = cv2.resize(frame, (128, 128))
        img_pil = Image.fromarray(img)
        input_img = transform(img_pil).unsqueeze(0).to(device)
        ouput = model(input_img)
        _, curr_char = torch.max(ouput, 1)
        curr_char = curr_char.detach().cpu().numpy()
        if prev_char != curr_char[0]:
            ouput_string += target_names[curr_char[0]]
            ouput_string += " "
            print(ouput_string)
            prev_char = curr_char[0]
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break