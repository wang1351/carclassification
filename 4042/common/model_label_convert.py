import pickle



path_to_file = './test.txt'
file = open(path_to_file,'r')
processed_label = []
model_dict = {}
for each in file:
    info = each.split('/')
    carmodel_label = int(info[1])
    if carmodel_label not in processed_label:
        processed_label.append(carmodel_label)

for i in range(len(processed_label)):
    model_dict[processed_label[i]] = i

dict_file = open('model_label_dict.pickle', 'wb')
pickle.dump(model_dict, dict_file)
dict_file.close()
