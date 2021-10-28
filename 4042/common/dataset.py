import torch
import cv2
import os
import pickle
import pdb

class CarMake_Classification(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_car = '/data/image/'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        path_to_img = path_to_img[:-1]
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label


    def __len__(self):
        return len(self.samples)

    def _init(self):
        #pdb.set_trace()
        carmake_file = open(self.split_file,'r')
        for each in carmake_file:
            # 78/1/2014/3ac218c0c6c378.jpg
            info = each.split('/')
            carmake_label = int(info[0])-1
            year = info[2]
            self.samples.append((os.path.join(self.path_to_car, each), carmake_label))
        carmake_file.close()


class CarModel_Classification(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_car = '/data/image/'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        path_to_img = path_to_img[:-1]
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label


    def __len__(self):
        return len(self.samples)

    def _init(self):
        #pdb.set_trace()
        carmake_file = open(self.split_file,'r')

        model_file = open('./model_label_dict.pickle', 'rb')
        model_label_dict = pickle.load(model_file)

        for each in carmake_file:
            info = each.split('/')
            carmodel_label = model_label_dict[int(info[1])]
            self.samples.append((os.path.join(self.path_to_car, each), carmodel_label))
        carmake_file.close()


class CarType_Classification(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_car = '/data/image/'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        path_to_img = path_to_img[:-1]
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label


    def __len__(self):
        return len(self.samples)

    def _init(self):
        #pdb.set_trace()
        carmake_file = open(self.split_file,'r')

        type_file = open('./car_type_dict.pickle', 'rb')
        type_label_dict = pickle.load(type_file)

        for each in carmake_file:
            info = each.split('/')
            cartype_label = int(info[1])
            if cartype_label in type_label_dict.keys():
                cartype_label = type_label_dict[int(info[1])]
                self.samples.append((os.path.join(self.path_to_car, each), cartype_label))
        carmake_file.close()


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_car = '/data/image/'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        path_to_img = path_to_img[:-1]
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label


    def __len__(self):
        return len(self.samples)

    def _init(self):
        #pdb.set_trace()
        triplet_file = open(self.split_file,'r')
        for each in triplet_file:
            info = each.split(' ')
            anchor = info[0]
            year = info[2]



            self.samples.append((os.path.join(self.path_to_car, each), carmake_label))
        carmake_file.close()
