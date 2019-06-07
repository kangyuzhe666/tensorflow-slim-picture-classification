f = open('labels.txt', 'a')
for i in range(101):
    f.writelines(str(i)+"\n")

f.close()