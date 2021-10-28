import torchvision.transforms as transforms
import torch.optim as optim
import json
import pdb

import torch.nn as nn
from settings import process_args
# from models import ResNet18
from alexnet import AlexNet
from dataset import *


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = process_args()
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((600, 420)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    classification_test = CarMake_Classification(split_file='./split_files/carmake_3split_test.txt', transforms=test_transform)

    # model = ResNet18(n_classes=431)
    # PATH_TO_MODEL = './check_points/1111_4pm_carmodel_resnet18/best_so_far.pth'
    model = AlexNet(num_classes=163)

    PATH_TO_MODEL = os.path.join(args['check_point_out_dir'],'CompCar_Res18','best_so_far.pth')

    weights = torch.load(PATH_TO_MODEL,map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(weights)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    testloader = torch.utils.data.DataLoader(classification_test, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss=0
        test_acc = []
        test_loss = []
        for i, data in enumerate(testloader):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)

            preds = torch.argmax(outputs, dim=1)

            for idx in range(len(labels)):
                if labels[idx] == preds[idx]:
                    correct += 1
                total += 1

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if (i % args['log_freq']) == 0:
                acc = correct / total
                test_loss.append(loss.item())
                test_acc.append(acc)
                print(' Iteration: {:03d}/{:03d} \t Loss: {:.05f}  \t Acc: {:.05f}'.format(i + 1,len(testloader),running_loss / (i + 1),acc))

            running_loss = 0
        with open('carType_test_acc.txt', 'w') as f:
            for item in test_acc:
                f.write("%s\n" % item)

        with open('carType_test_loss.txt', 'w') as f:
            for item in test_loss:
                f.write("%s\n" % item)











