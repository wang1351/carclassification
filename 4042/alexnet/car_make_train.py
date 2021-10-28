import torchvision.transforms as transforms
import torch.optim as optim
import json
import pdb

import torch.nn as nn
from settings import process_args
#from models import ResNet18
#import torchvision.models as models
from alexnet import AlexNet
#alexnet = models.alexnet()
from dataset import *


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.backends.cudnn.benchmark = True


def train_epoch(model, current_epoch, criterion, optimizer, dataset, args):
    running_loss = 0
    total_loss = 0
    correct = 0
    total = 0
    train_acc = []
    train_loss = []
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    num_iters = len(trainloader)
    for i, data in enumerate(trainloader):
        input, labels = data
        input = input.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)

            for idx in range(len(labels)):
                if labels[idx] == preds[idx]:
                    correct += 1
                total += 1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()


        if (i % args['log_freq']) == 0:

            acc = correct/total
            train_loss.append(loss.item())
            train_acc.append(acc)
            print('Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Loss: {:.05f}  \t Acc: {:.05f}'.format(current_epoch,
                                                                                                       i + 1,
                                                                                                       num_iters,
                                                                                                       running_loss / (
                                                                                                               i + 1),
                                                                                                    acc))

        running_loss = 0
    with open('carmake_train_epoch_'+str(current_epoch)+'_acc.txt', 'w') as f:
        for item in train_acc:
            f.write("%s\n" % item)

    with open('carmake_train_epoch_'+str(current_epoch)+'_loss.txt', 'w') as f:
        for item in train_loss:
            f.write("%s\n" % item)

    return total_loss/num_iters, model

def val_epoch(model, current_epoch ,criterion, val_dataset, args):
    with torch.no_grad():
        total_loss = 0
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                                 num_workers=0)
        total = 0
        running_loss = 0
        num_iters = len(val_loader)
        correct = 0
        val_acc=[]
        for i, data in enumerate(val_loader):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)
            loss = criterion(outputs, labels)

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                for idx in range(len(labels)):
                    if labels[idx] == preds[idx]:
                        correct += 1
                    total += 1

            total_loss += loss.item()
            running_loss += loss.item()

            if (i % args['log_freq']) == 0:
                acc = correct/total
                val_acc.append(acc)
                print('Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Loss: {:.05f}  \t Acc: {:.05f}'.format(current_epoch, i + 1,
                                                                                                           num_iters,
                                                                                                           running_loss / (
                                                                                                                       i + 1),
                                                                                                           acc))
        with open('carmake_val_epoch_'+str(current_epoch)+'_acc.txt', 'w') as f:
            for item in val_acc:
                f.write("%s\n" % item)


    return total_loss/num_iters, correct/num_iters


if __name__ == '__main__':
    args = process_args()
    if not os.path.exists(os.path.join(args['check_point_out_dir'], args['exp_name'])):
        os.makedirs(os.path.join(args['check_point_out_dir'], args['exp_name']))
    if not os.path.exists(args['runargs_out_dir']):
        os.makedirs(args['runargs_out_dir'])
    with open(os.path.join(args['runargs_out_dir'], '{:s}.json'.format(args['exp_name'])), 'wt') as fp:
        json.dump(args, fp, indent=2)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((600, 420)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((600, 420)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    classification_train = CarMake_Classification(split_file='./split_files/carmake_3plit_train.txt', transforms=train_transform)

    classification_val = CarMake_Classification(split_file='./split_files/carmake_3split_val.txt', transforms=val_transform)

    current_best_val_acc = 0
    model = AlexNet(num_classes=163)
    #model = models.alexnet(num_classes=163)
    PATH_TO_MODEL = os.path.join(args['check_point_out_dir'],'CompCar_Res18','best_so_far.pth')
    weights = torch.load(PATH_TO_MODEL)['model_state_dict']
    model.load_state_dict(weights)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['moment'])

    for epoch in range(args['num_epoch']):
        train_loss, model = train_epoch(model, epoch, criterion, optimizer, classification_train, args)
        val_loss, val_acc = val_epoch(model, epoch, criterion, classification_val, args)

        if val_acc > current_best_val_acc:
            current_best_val_acc = val_acc
            name = 'best_so_far.pth'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                        }, os.path.join(args['check_point_out_dir'], args['exp_name'], name))
            print('Model checkpoint written! :{:s}'.format(name))










