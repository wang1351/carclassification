import torch
import cv2
import os
import random
import pickle
import pdb
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_car = '/Users/wzy/Downloads/CompCars/data/image/'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self.dict = {} #dictionary to select positive and negative path
        self._init()

    def __getitem__(self, index):
        path_to_anchor, label = self.samples[index]

        negative_label = 0
        #negative_label should not be the same as label
        while negative_label == label:
            negative_label = random.randint(0, 430)

        #find path to positive
        path_to_positive = self.dict[label][random.randint(0, len(self.dict[label]) - 1)]
        while path_to_positive == path_to_anchor:
            path_to_positive = self.dict[label][random.randint(0, len(self.dict[label]) - 1)]

        path_to_negative = self.dict[negative_label][random.randint(0, len(self.dict[negative_label])-1)]
        # pdb.set_trace()

        anchor, positive, negative = cv2.imread(path_to_anchor[:-1]), cv2.imread(path_to_positive[:-1]), cv2.imread(path_to_negative[:-1])
        anchor, positive, negative = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB), cv2.cvtColor(positive, cv2.COLOR_BGR2RGB), cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
        if self.transforms:
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

        return anchor, positive, negative, label


    def __len__(self):
        return len(self.samples)

    def _init(self):
        #pdb.set_trace()
        carmodel_file = open(self.split_file,'r')
        model_file = open('./model_label_dict.pickle', 'rb')
        model_label_dict = pickle.load(model_file)
        for each in carmodel_file:
            info = each.split('/')
            carmodel_label = model_label_dict[int(info[1])]
            self.samples.append((os.path.join(self.path_to_car, each), carmodel_label))

            if carmodel_label not in self.dict.keys():
                self.dict[carmodel_label] = []
            self.dict[carmodel_label].append(os.path.join(self.path_to_car, each))


        carmodel_file.close()

