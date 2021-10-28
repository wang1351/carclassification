import os
import numpy as np

SEED = 2
np.random.seed(SEED)

train = []
val = []
test=[]


#78/1/2014/3ac218c0c6c378.jpg
#path = '/Users/wzy/Downloads/CompCars/data/image'
path = '/data/image'
for carmake_label in os.listdir(path):
    this_carmake = []
    selected = []
    i = 0
    for carmodel_label in os.listdir(os.path.join(path,carmake_label)):
        for year in os.listdir(os.path.join(path,carmake_label,carmodel_label)):
            for id in os.listdir(os.path.join(path, carmake_label, carmodel_label, year)):
                inst = '/'.join([str(carmake_label), str(carmodel_label), str(year), str(id)])
                this_carmake.append(inst)

    idx = np.arange(len(this_carmake))
    np.random.shuffle(idx)
    while i < 70 and i < len(this_carmake):  #change this number to change sample size!
        selected.append(this_carmake[idx[i]])
        i += 1



    for each in selected:
        a = np.random.rand()
        if a < 0.7:
            train.append(each)
        elif a < 0.8:
            val.append(each)
        else :
            test.append(each)


writer = open('carmake_3plit_train.txt', 'w')
for name in train:
    writer.write(name + '\n')
writer.close()

writer = open('carmake_3split_val.txt', 'w')
for name in val:
    writer.write(name + '\n')
writer.close()

writer = open('carmake_3split_test.txt', 'w')
for name in test:
    writer.write(name + '\n')
writer.close()


