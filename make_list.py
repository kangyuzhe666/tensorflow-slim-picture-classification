import os

class_names_to_ids = {}
file = []
count = 0
data_dir = 'images/'
output_path = 'list.txt'
fd = open(output_path, 'w')

for root, dirs, files in os.walk(data_dir):
    file = dirs
    break

for i in file:
    class_names_to_ids.setdefault(i,count)
    count = count + 1
print(class_names_to_ids)
print("total: ",count," class")
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()