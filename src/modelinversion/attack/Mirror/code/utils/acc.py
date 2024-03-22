import torch
from torch import nn

def verify_acc(inputs, labels, model, arch_name):
    
    device = inputs.device
    
    acc = 0
    
    with torch.no_grad():
    
        pred = model(inputs)
        if arch_name == 'sphere20a':
            pred = pred.result
        else: 
            pred = pred.result
            
        confidence = nn.functional.softmax(pred, dim=-1)
        pred_label = torch.argmax(pred, dim=-1)
        acc = (pred_label == labels).sum() / len(labels)

    return acc