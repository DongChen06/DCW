import cv2
import os
from os import walk
import random
import copy
from aug_utils import readYolo, writeYolo, getTransform
from draw_boxes import draw_boxes
from shutil import copyfile


# set random seed here
random.seed(66)
images_path = '../datasets/Dataset_final/'
debug = False  # plot the augmented images with bounding boxes


# we only augment the images in the "train" folder
def aug_images(folder_idx):
    images_dir = images_path + 'DATA_' + str(folder_idx) + '/train/images'
    label_dir = images_path + 'DATA_' + str(folder_idx) + '/train/labels/'
    images_out_dir = images_path + 'DATA_' + str(folder_idx) + '/train/images_aug/'
    labels_out_dir = images_path + 'DATA_' + str(folder_idx) + '/train/labels_aug/'
    debug_path = images_path + 'DATA_' + str(folder_idx) + '/train/images_debug_aug/'

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    files = []
    ext = (".jpeg", ".jpg", ".png", ".PNG")

    print("Image Folders: ", images_dir)
    for (dirpath, dirnames, filenames) in walk(images_dir):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(dirpath, filename))

    print("Working...")
    for file in files:
        print(file)

        image = cv2.imread(file)
        xmlTitle, txtExt = os.path.splitext(os.path.basename(file))
        cv2.imwrite(os.path.join(images_out_dir, os.path.basename(file)), image)
        copyfile(label_dir + xmlTitle + '.txt', labels_out_dir + xmlTitle + '.txt')

        # start the augmentation
        bboxes = readYolo(label_dir + xmlTitle + '.txt')

        # each image is augmented by 3 times
        for i in range(0, 3):
            img = copy.deepcopy(image)
            transform = getTransform(random.randrange(10))  # we have 10 augmentation methods
            transformed = transform(image=img, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            # for debug use
            if debug:
                annot_image, box_areas = draw_boxes(transformed_image, transformed_bboxes, 'yolo')
                cv2.imwrite(debug_path + xmlTitle + '_' + str(i) + '.jpg', annot_image)

            cv2.imwrite(images_out_dir + xmlTitle + '_' + str(i) + '.jpg', transformed_image)
            label_out_name = labels_out_dir + xmlTitle + '_' + str(i) + '.txt'
            writeYolo(transformed_bboxes, label_out_name)


if __name__ == "__main__":

    for i in range(3):
        aug_images(i)