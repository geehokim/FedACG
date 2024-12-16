import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

__all__ = [ 'get_numclasses','count_label_distribution','check_data_distribution', 'get_optimizer', 'get_scheduler','modeleval','create_pth_dict']


def count_label_distribution(labels,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,label in enumerate(labels):
        data_distribution[label]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images,target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def get_numclasses(args,trainset = None):
    if args.dataset.name in ['CIFAR10', "MNIST"]:
        num_classes=10
    elif args.dataset.name in ["CIFAR100"]:
        num_classes=100
    elif args.dataset.name in ["TinyImageNet"]:
        num_classes=200
    elif args.dataset.name in ["iNaturalist"]:
        num_classes=1203
    elif args.dataset.name in ["ImageNet"]:
        num_classes=1000
    elif args.dataset.name in ["leaf_celeba"]:
        num_classes = 2
    elif args.dataset.name in ["leaf_femnist"]:
        num_classes = 62
    elif args.set in ["shakespeare"]:
        num_classes=80
    else:
        assert False
        
    print("number of classes in ", args.dataset.name," is : ", num_classes)
    return num_classes

def get_optimizer(args, parameters):
    if args.set=='CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="MNIST":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="CIFAR100":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Invalid mode")
        return
    return optimizer

def get_scheduler(optimizer, args):
    if args.set=='CIFAR10':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)  
    elif args.set=="CIFAR100":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        
    else:
        print("Invalid mode")
        return
    return scheduler

def modeleval(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / float(total)))
    acc = (100 * correct / float(total))
    model.train()
    return acc

def get_prefix_idx(x):
    idx = -4
    while True:
        try:
            int(x[idx-1])
            idx-=1
        except:
            break
    return idx


def get_prefix_num(x):
    idx = get_prefix_idx(x)
    return x[:idx], int(x[idx:-4])




def create_pth_dict(pth_path):
    pth_dir = os.path.dirname(pth_path)
    pth_base = os.path.basename(pth_path)
    pth_prefix,_ = get_prefix_num(pth_base)

    pth_dict = {}

    for filename in os.listdir(pth_dir):
        
        if filename.startswith(pth_prefix):
            _,number = get_prefix_num(filename)
            filepath = os.path.join(pth_dir, filename)
            pth_dict[number] = filepath

    return dict(sorted(pth_dict.items()))





