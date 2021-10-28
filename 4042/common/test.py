import pickle

list3_file = open('../split_files/carmake_3split_val.txt', 'r')
a = []
for each in list3_file:

    b = each.split('/')[0]
    if b not in a:
        a.append(b)
print(len(a))
