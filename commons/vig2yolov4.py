import cv2
import os
from os import walk
import json
import sys

sys.path.append("../")

# image_roots = ["datasets/09032021_MEFAS_Brooksville",
#               "datasets/09032021_MEFAS_WBAndrews", "datasets/09042021_MEFAS_WBAndrews",
#               "datasets/09062021_MEFAS_WBAndrews", "datasets/09082021_MEFAS_WBAndrews",
#               "datasets/09092021_MEFAS_WBAndrews", "datasets/09102021_MEFAS_WBAndrews",
#                "datasets/09122021_MEFAS_WBAndrews",  "datasets/09132021_MEFAS_NorthFarm"]
# image_roots = ["datasets/09042021_MEFAS_WBAndrews"]
image_roots = ["datasets/08272021_MEFAS_BROOKSVILLE"]

out_dir = "datasets/CottonWeedYolov4"
os.makedirs(out_dir, exist_ok=True)
type_folder = "val"
os.makedirs(out_dir + '/' + type_folder, exist_ok=True)

files = []
ext = (".jpeg", ".jpg", ".png", ".PNG")

for image_root in image_roots:
    print("Image Folders: ", image_root)
    for (dirpath, dirnames, filenames) in walk(image_root):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(dirpath, filename))

idx = 0
length = 0
json_path = 'commons/class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
json_file = open(json_path, "r")
class_dict = json.load(json_file)

print("Working...")
for idx_img, file in enumerate(files):
    length += 1
    if length % 1000 == 0:
        print("Working on the {}th image...".format(length))

    img = cv2.imread(file)
    img_path = file.split("/")[-1][:-4] + "-" + str(idx_img) + '.jpg'
    cv2.imwrite(os.path.join(out_dir, type_folder, img_path), img)

    # read the corresponding json files
    json_file1 = open(file.split('.')[0] + '.json', "r")
    label_indict = json.load(json_file1)
    txt_path = os.path.join(out_dir, type_folder + '.txt')

    if isinstance(list(label_indict.values())[0]['regions'], dict):
        region = list(label_indict.values())[0]['regions']
        x1 = region['shape_attributes']['x']
        y1 = region['shape_attributes']['y']
        x2 = x1 + region['shape_attributes']['width'] - 1
        y2 = y1 + region['shape_attributes']['height'] - 1
        img_path = file.split("/")[-1][:-4] + "-" + str(idx_img) + '.jpg'

        label = list(region['region_attributes']['CottonWeed'].keys())[0]
        if label == 'Unknown':
            # remove the image since it has only an "Unknown label"
            print("The image contains only an Unknown label: ", file)
            os.remove(os.path.join(out_dir, type_folder, img_path))
            length -= 1
        else:
            weed_class = class_dict[label]
            with open(txt_path, 'a', encoding='UTF-8') as f:
                f.write('{} {},{},{},{},{}'.format(img_path, x1, y1, x2, y2, weed_class))
                f.write('\n')
    else:
        regions = list(label_indict.values())[0]['regions']
        len_labels = len(regions)
        len_label = 0
        write_image_name = False
        img_path = file.split("/")[-1][:-4] + "-" + str(idx_img) + '.jpg'

        for idx, r in enumerate(regions):
            label = list(r['region_attributes']['CottonWeed'].keys())[0]
            if label == 'Unknown':
                # print("The image contains Unknown labels: ", file)
                len_label += 1

        if len_label == len_labels:
            print("The image contains only Unknown labels : ", file)
            os.remove(os.path.join(out_dir, type_folder, img_path))
            length -= 1
        else:
            for idx, r in enumerate(regions):
                x1 = r['shape_attributes']['x']
                y1 = r['shape_attributes']['y']
                x2 = x1 + r['shape_attributes']['width'] - 1
                y2 = y1 + r['shape_attributes']['height'] - 1

                label = list(r['region_attributes']['CottonWeed'].keys())[0]
                if label == 'Unknown':
                    # print("The image contains Unknown labels: ", file)
                    pass
                else:
                    weed_class = class_dict[label]
                    if idx == 0:
                        write_image_name = True
                        with open(txt_path, 'a', encoding='UTF-8') as f:
                            f.write('{} {},{},{},{},{}'.format(img_path, x1, y1, x2, y2, weed_class))
                    else:
                        if write_image_name:
                            with open(txt_path, 'a', encoding='UTF-8') as f:
                                f.write(' {},{},{},{},{}'.format(x1, y1, x2, y2, weed_class))
                        else:
                            write_image_name = True
                            with open(txt_path, 'a', encoding='UTF-8') as f:
                                f.write('{} {},{},{},{},{}'.format(img_path, x1, y1, x2, y2, weed_class))


            with open(txt_path, 'a', encoding='UTF-8') as f:
                f.write('\n')


print("----------Summary----------")
print("Overall images: ", length)
print("----------Summary----------")
