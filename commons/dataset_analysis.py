import cv2
import os, csv
from os import walk
import json
import sys

sys.path.append("../")

# store the # of (BB, #image) for each weed, #image represents the weeds occur in how many images
dic = {
    "Waterhemp": [0, 0],
    "MorningGlory": [0, 0],
    "Purslane": [0, 0],
    "SpottedSpurge": [0, 0],
    "Carpetweed": [0, 0],
    "Ragweed": [0, 0],
    "Eclipta": [0, 0],
    "PricklySida": [0, 0],
    "PalmerAmaranth": [0, 0],
    "Sicklepod": [0, 0],
    "Goosegrass": [0, 0],
    "CutleafGroundcherry": [0, 0],
    "Horsenettle": [0, 0],
    "Nutsedge": [0, 0],
    "HophornbeamCopperleaf": [0, 0],
    "Smellmelon": [0, 0],
    "PurpleAmmannia": [0, 0],
    "PricklyLettuce": [0, 0],
    "HempSesbania": [0, 0],
    "VirginiaButtonweed": [0, 0],
    "Cocklebur": [0, 0]}


image_roots = ["datasets/0809_0820_2021_MEFAS", "datasets/06282021_MEFAS_NorthFarm",
                "datasets/07122021_MEFAS_NorthFarm",
                "datasets/08272021_MEFAS_BROOKSVILLE", "datasets/09032021_MEFAS_Brooksville",
               "datasets/09032021_MEFAS_WBAndrews", "datasets/09042021_MEFAS_WBAndrews",
               "datasets/09062021_MEFAS_WBAndrews", "datasets/09082021_MEFAS_WBAndrews",
               "datasets/09092021_MEFAS_WBAndrews", "datasets/09102021_MEFAS_WBAndrews",
               "datasets/09122021_MEFAS_WBAndrews", "datasets/09132021_MEFAS_NorthFarm"]
# image_roots = ["datasets/09132021_MEFAS_NorthFarm"]

files = []
ext = (".jpeg", ".jpg", ".png", ".PNG")

for image_root in image_roots:
    print("Image Folders:", image_root)
    for (dirpath, dirnames, filenames) in walk(image_root):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(dirpath, filename))

if not os.path.isfile('dataset_analysis.csv'):
    with open('dataset_analysis.csv', mode='w') as csv_file:
        fieldnames = ['Image Path', "Width", "Height", "Weeds", "x", "y", "w", "h"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

length = 0
max_bb = 1  # maximum # of bb
max_bb_image = None
total_bb = 0

print("Working...")
for file in files:
    length += 1
    if length % 1000 == 0:
        print("Working on the {}th image...".format(length))

    img = cv2.imread(file)
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]

    # read the corresponding json files
    json_file1 = open(file.split('.')[0] + '.json', "r")
    label_indict = json.load(json_file1)

    if isinstance(list(label_indict.values())[0]['regions'], dict):
        # for image containing only one weed class
        region = list(label_indict.values())[0]['regions']
        x_min = region['shape_attributes']['x']
        y_min = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        label = list(region['region_attributes']['CottonWeed'].keys())[0]
        if label == 'Unknown':
            length -= 1
        else:
            dic[label][0] += 1
            dic[label][1] += 1
            total_bb += 1
            with open('dataset_analysis.csv', 'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow([file.split('/')[-1], width, height, label, x_min, y_min, w, h])
    else:
        # for image containing multiple weed classes
        regions = list(label_indict.values())[0]['regions']
        len_labels = len(regions)
        len_unknown_label = 0
        weeds = []

        for r in regions:
            x_min = r['shape_attributes']['x']
            y_min = r['shape_attributes']['y']
            w = r['shape_attributes']['width']
            h = r['shape_attributes']['height']

            label = list(r['region_attributes']['CottonWeed'].keys())[0]
            if label == 'Unknown':
                # print("The image contains Unknown labels: ", file)
                len_unknown_label += 1

            if label != 'Unknown':
                dic[label][0] += 1
                if label not in weeds:
                    dic[label][1] += 1
                    weeds.append(label)

                total_bb += 1
                with open('dataset_analysis.csv', 'a+', newline='') as write_obj:
                    csv_writer = csv.writer(write_obj)
                    csv_writer.writerow([file.split('/')[-1], width, height, label, x_min, y_min, w, h])

        if len_labels - len_unknown_label > max_bb:
            max_bb = len_labels - len_unknown_label
            max_bb_image = file

        if len_unknown_label == len_labels:
            length -= 1

print("----------Summary----------")
print("Overall images:", length)
print("Total number of bounding boxes:", total_bb)
print("Maximum number of bounding boxes:", max_bb)
print("The image has the maximum number of bounding boxes:", max_bb_image)
print("Bounding Boxes:", dic)
print("----------Summary----------")

with open('bounding_box_summary.json', 'w') as fp:
    json.dump(dic, fp)
