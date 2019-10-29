import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 43 # GTSRB as 43 classes
    
#### MODEL 1 : MOBILENET ####

model_mobilenet = models.mobilenet_v2()

## Create a custom classifer with 43(num of GTSB classes) outputs. 
custom_classifier = nn.Sequential(OrderedDict([
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc1', nn.Linear(1280, nclasses)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model_mobilenet.classifier = custom_classifier

## Load parameters from the checkpoint file. The parameters loaded here are trained from scratch.
state_dict = torch.load('./mobilenet.pth', map_location=torch.device(device))
model_mobilenet.load_state_dict(state_dict)


#### MODEL 2 : DENSENET ####

model_densenet = models.densenet121()

custom_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 43)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model_densenet.classifier = custom_classifier

state_dict = torch.load('./densenet.pth', map_location=torch.device(device))
model_dense.load_state_dict(state_dict)


#### MODEL 3 : ENSEMBLE ####

class MyEnsemble(nn.Module):
    def __init__(self, m1, m2):
        super(MyEnsemble, self).__init__()
        self.model1 = m1
        self.model2 = m2
        self.avg_weight = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = self.avg_weight[0]*out1 + (1-self.avg_weight[0])*out2
        x = F.log_softmax(out, dim=1)
        return x

model = MyEnsemble(model_dense, model_mobilenet)

## Freeze the weights of the individual models. Only change the prediction averaging weight.
for pid, param in enumerate(model.parameters()):
    if pid != 0:
        param.requires_grad = False
        
optimizer = optim.Adam([list(model.parameters())[0]], lr=0.001)
