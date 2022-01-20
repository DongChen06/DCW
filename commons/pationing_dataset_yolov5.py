"""
Partition dataset of images into training, validation and testing sets
"""
import os
from shutil import copyfile
import argparse
import math
import random
from os import walk


def iterate_dir(source, labelDir, labeljsonDir, dest, ratio_list):
    # generate train, val and test dataset
    os.makedirs(os.path.join(dest, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'train', 'labels_json'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'labels_json'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'test', 'labels_json'), exist_ok=True)

    train_txt_path = os.path.join(dest, 'train' + '.txt')
    val_txt_path = os.path.join(dest, 'val' + '.txt')
    test_txt_path = os.path.join(dest, 'test' + '.txt')

    # get all the pictures in directory
    images = []
    ext = (".JPEG", "jpeg", "JPG", ".jpg", ".png", "PNG")

    for (dirpath, dirnames, filenames) in walk(source):
        for filename in filenames:
            if filename.endswith(ext):
                images.append(os.path.join(dirpath, filename))

    num_images = len(images)
    num_val_images = math.ceil(ratio_list[1] * num_images)
    num_test_images = math.ceil(ratio_list[2] * num_images)

    print("total images", "n_train", "n_val", "n_test",
          num_images, (num_images - num_val_images - num_test_images), num_val_images, num_test_images)

    for j in range(num_val_images):
        idx = random.randint(0, len(images) - 1)
        filename = images[idx].split("/")[-1]
        filename_labels = images[idx].split("/")[-1][:-4] + '.txt'
        filename_labels_json = images[idx].split("/")[-1][:-4] + '.json'
        copyfile(os.path.join(source, filename),
                 os.path.join(os.path.join(dest, 'val', 'images'), filename))
        copyfile(os.path.join(labelDir, filename_labels),
                 os.path.join(os.path.join(dest, 'val', 'labels'), filename_labels))
        copyfile(os.path.join(labeljsonDir, filename_labels_json),
                 os.path.join(os.path.join(dest, 'val', 'labels_json'), filename_labels_json))
        images.remove(images[idx])

        with open(val_txt_path, 'a', encoding='UTF-8') as f:
            f.write('{}'.format(os.path.join(os.path.join(dest, 'val', 'images'), filename)))
            f.write('\n')

    for i in range(num_test_images):
        idx = random.randint(0, len(images) - 1)
        filename = images[idx].split("/")[-1]
        filename_labels = images[idx].split("/")[-1][:-4] + '.txt'
        filename_labels_json = images[idx].split("/")[-1][:-4] + '.json'
        copyfile(os.path.join(source, filename),
                 os.path.join(os.path.join(dest, 'test', 'images'), filename))
        copyfile(os.path.join(labelDir, filename_labels),
                 os.path.join(os.path.join(dest, 'test', 'labels'), filename_labels))
        copyfile(os.path.join(labeljsonDir, filename_labels_json),
                 os.path.join(os.path.join(dest, 'test', 'labels_json'), filename_labels_json))
        images.remove(images[idx])

        with open(test_txt_path, 'a', encoding='UTF-8') as f:
            f.write('{}'.format(os.path.join(os.path.join(dest, 'test', 'images'), filename)))
            f.write('\n')

    for file in images:
        filename = file.split("/")[-1]
        filename_labels = file.split("/")[-1][:-4] + '.txt'
        filename_labels_json = file.split("/")[-1][:-4] + '.json'
        copyfile(os.path.join(source, filename),
                 os.path.join(os.path.join(dest, 'train', 'images'), filename))
        copyfile(os.path.join(labelDir, filename_labels),
                 os.path.join(os.path.join(dest, 'train', 'labels'), filename_labels))
        copyfile(os.path.join(labeljsonDir, filename_labels_json),
                 os.path.join(os.path.join(dest, 'train', 'labels_json'), filename_labels_json))

        with open(train_txt_path, 'a', encoding='UTF-8') as f:
            f.write('{}'.format(os.path.join(os.path.join(dest, 'train', 'images'), filename)))
            f.write('\n')


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets")
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default='/home/dong9/PycharmProjects/CottonWeed_Detection/datasets/CottonWeedDataYolov5/images')
    parser.add_argument(
        '-l', '--labelDir',
        help='Path to the folder where the labels are stored. If not specified, the CWD will be used.',
        type=str,
        default='/home/dong9/PycharmProjects/CottonWeed_Detection/datasets/CottonWeedDataYolov5/labels')
    parser.add_argument(
        '-lj', '--labeljsonDir',
        help='Path to the folder where the json labels are stored. If not specified, the CWD will be used.',
        type=str,
        default='/home/dong9/PycharmProjects/CottonWeed_Detection/datasets/CottonWeedDataYolov5/labels_json')
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the yolov4 folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default='/home/dong9/PycharmProjects/CottonWeed_Detection/datasets/CottonWeedDataYolov5')
    parser.add_argument(
        '-r', '--ratio_list',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=[0.65, 0.2, 0.15],
        type=list)
    args = parser.parse_args()

    # for i in range(5):
    #     random.seed(i)
    #     outputDir = args.outputDir + '/DATA_{}'.format(i)
    #
    #     # Now we are ready to start the iteration
    #     iterate_dir(args.imageDir, outputDir, args.ratio_list)

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.labelDir, args.labeljsonDir, args.outputDir, args.ratio_list)


if __name__ == '__main__':
    main()
