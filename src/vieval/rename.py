import os

dir = "./results"
# dir = './results'

filenames = os.listdir(dir)

for name in filenames:
    path = os.path.join(dir, name)
    os.rename(path, path.replace(" ", ""))
