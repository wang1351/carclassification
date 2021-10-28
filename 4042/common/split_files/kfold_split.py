import os
import numpy as np

#5fold example
folds = [[]for i in range(5)]
test=[]


#78/1/2014/3ac218c0c6c378.jpg
path = '/Users/wzy/Downloads/CompCars/data/image'
for carmake_label in os.listdir(path):
    for carmodel_label in os.listdir(os.path.join(path,carmake_label)):
        for year in os.listdir(os.path.join(path,carmake_label,carmodel_label)):
            for id in os.listdir(os.path.join(path, carmake_label, carmodel_label, year)):
                inst = '/'.join([str(carmake_label), str(carmodel_label), str(year), str(id)])
                a = np.random.rand()
                if a < 0.16:
                    folds[0].append(inst)
                elif a < 0.32:
                    folds[1].append(inst)
                elif a < 0.48:
                    folds[2].append(inst)
                elif a < 0.64:
                    folds[3].append(inst)
                elif a < 0.8:
                    folds[4].append(inst)
                else:
                    test.append(inst)

for i in range(1,6):
    writer = open('5fold_'+str(i)+'.txt', 'w')
    for inst in folds[i-1]:
        writer.write(inst + '\n')
    writer.close()


writer = open('5fold_test.txt', 'w')
for name in test:
    writer.write(name + '\n')
writer.close()


