import albumentations as A
import cv2
import os
from xml.dom import minidom
import random
import copy
from draw_boxes import draw_boxes


imagespath = '/home/dong9/PycharmProjects/DCW/datasets/Data_aug/'
output_dir = '/home/dong9/PycharmProjects/DCW/datasets/Data_augmented/'
output_dir_box = '/home/dong9/PycharmProjects/DCW/datasets/Data_augmented_box/'
random.seed(7)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_box, exist_ok=True)


def readImage(filename):
    # OpenCV uses BGR channels
    img = cv2.imread(imagespath+filename)
    return img


def getCoordinates(filename):
    allbb = []
    xmldoc = minidom.parse(imagespath+filename)
    itemlist = xmldoc.getElementsByTagName('object')

    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    for item in itemlist:
        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        xmin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymax')[0]).firstChild.data

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        b = [xmin, ymin, xmax, ymax, classid]
        allbb.append(b)
    return allbb


def start():
    for filename in sorted(os.listdir(imagespath)):

        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            title, ext = os.path.splitext(os.path.basename(filename))
            image = readImage(filename)

        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = readYolo(imagespath+xmlTitle+'.txt')
                for i in range(0, 11):
                    img = copy.deepcopy(image)
                    transform = getTransform(i)
                    try:
                        transformed = transform(image=img, bboxes=bboxes)
                        transformed_image = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        name = title + str(i) + '.jpg'

                        annot_image, box_areas = draw_boxes(transformed_image, transformed_bboxes, 'yolo')
                        annot_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_dir_box + name, annot_image)

                        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_dir + name, transformed_image)
                        # print(transformed_bboxes)
                        # writeVoc(transformed_bboxes, count, transformed_image)
                        writeYolo(transformed_bboxes, i, title)
                    except:
                        print("bounding box issues")
                        pass


def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            coords.append(x)
    return coords


def writeYolo(coords, count, name):

    with open(output_dir + name+str(count)+'.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))


def getTransform(loop):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.5, p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 2:
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=0.5, p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 3:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 4:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 5:
        transform = A.Compose([
            A.JpegCompression(quality_lower=0, quality_upper=1, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 6:
        transform = A.Compose([
            A.FancyPCA(alpha=0.6, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 7:
        transform = A.Compose([
            A.Blur(blur_limit=30, p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 8:
        transform = A.Compose([
            A.GaussNoise(var_limit=(0, 100.0), p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 9:
        transform = A.Compose([
            A.RGBShift(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 10:
        transform = A.Compose([
            A.RGBShift(p=0)
        ], bbox_params=A.BboxParams(format='yolo'))

    return transform

start()