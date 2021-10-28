import pickle

model_file = open('./attributes.txt', 'rb')
car_type_dict = {}

for each in model_file:
    info = str(each).split(' ')
    car_type = int(info[-1][0]) - 1
    model = int(info[0].split("'")[-1])
    print(info)
    print(model, car_type)
    if car_type >= 0:
        car_type_dict[model] = car_type

dict_file = open('car_type_dict.pickle', 'wb')
pickle.dump(car_type_dict, dict_file)
dict_file.close()