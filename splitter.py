import os
import glob
import random
import re
import cv2

root = "nirscene1"

folderReg = os.path.join(root, "*")
allFolders = glob.glob(folderReg)
allFiles = {}

for dir in allFolders:
    dirName = os.path.basename(dir)
    allFiles[dirName] = {}
    rgbPath = os.path.join(dir, "*rgb*")
    allFiles[dirName] = glob.glob(rgbPath)
# print(allFiles)
counter = 1
tCounter = 1
container = {}
testFiles = {}
# import pdb
# pdb.set_trace()
for dir in allFiles:
    dirPath = os.path.join(root, dir)
    nirReg = re.compile(r"(\d+)_rgb")
    for idx in range(4):
        rgbFile = allFiles[dir].pop(random.randrange(len(allFiles[dir])))
        match = re.search(nirReg, rgbFile)
        number = match.group(1)
        nirPath = os.path.join(dirPath, "{}_nir*".format(number))
        nirFile = glob.glob(nirPath)
        testFiles[tCounter] = [rgbFile, nirFile[0]]
        tCounter += 1

    for rgbFile in allFiles[dir]:
        match = re.search(nirReg, rgbFile)
        number = match.group(1)
        nirPath = os.path.join(dirPath, "{}_nir*".format(number))
        nirFile = glob.glob(nirPath)
        container[counter] = [rgbFile, nirFile[0]]
        counter += 1
import pdb
pdb.set_trace()

for counter in container:
    rgb, nir = container[counter]
    rgbImg = cv2.imread(rgb)
    nirImg = cv2.imread(nir, 0)
    rgbImg = cv2.resize(rgbImg, (224, 224))
    nirImg = cv2.resize(nirImg, (224, 224))
    rgbPath = os.path.join("dataset", "rgb", "{}.png".format(counter))
    nirPath = os.path.join("dataset", "nir", "{}.png".format(counter))
    cv2.imwrite(rgbPath, rgbImg)
    cv2.imwrite(nirPath, nirImg)

for counter in testFiles:
    rgb, nir = testFiles[counter]
    rgbImg = cv2.imread(rgb)
    nirImg = cv2.imread(nir, 0)
    rgbImg = cv2.resize(rgbImg, (224, 224))
    nirImg = cv2.resize(nirImg, (224, 224))
    rgbPath = os.path.join("dataset", "test_rgb", "{}.png".format(counter))
    nirPath = os.path.join("dataset", "test_nir", "{}.png".format(counter))
    cv2.imwrite(rgbPath, rgbImg)
    cv2.imwrite(nirPath, nirImg)


print(container)
print(len(container.keys()))





