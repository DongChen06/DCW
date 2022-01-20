from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import xml.dom.minidom
import os
import sys
from PIL import Image
import json


# 把txt中的内容写进xml
def deal(path):
    files = os.listdir(path)  # 列出所有文件
    for file in files:
        filename = os.path.splitext(file)[0]  # 分割出文件名
        # print(filename)

        sufix = os.path.splitext(file)[1]  # 分割出后缀
        # import pdb
        # pdb.set_trace()
        p_file = path + "/" + file
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        names = []
        num = 0
        with open(p_file) as fp:
            json_data = json.load(fp)
            regions = json_data[[k for k in json_data][0]]['regions']
            if (type(regions).__name__ == 'dict'):
                shape_attributes = regions['shape_attributes']
                xmin = shape_attributes['x']
                ymin = shape_attributes['y']
                xmax = xmin + regions['shape_attributes']['width'] - 1
                ymax = ymin + regions['shape_attributes']['height'] - 1
                name = [x for x in regions['region_attributes']['CottonWeed']][0]
                xmins.append(xmin)
                ymins.append(ymin)
                xmaxs.append(xmax)
                ymaxs.append(ymax)
                names.append(name)
                num = 1
            elif (type(regions).__name__ == 'list'):
                num = len(regions)
                for obj in regions:
                    shape_attributes = obj['shape_attributes']
                    xmin = shape_attributes['x']
                    ymin = shape_attributes['y']
                    xmax = xmin + obj['shape_attributes']['width'] - 1
                    ymax = ymin + obj['shape_attributes']['height'] - 1
                    name = [x for x in obj['region_attributes']['CottonWeed']][0]
                    xmins.append(xmin)
                    ymins.append(ymin)
                    xmaxs.append(xmax)
                    ymaxs.append(ymax)
                    names.append(name)
        # pdb.set_trace()
        print('done')

        if sufix == '.json':
            # num, xmins, ymins, xmaxs, ymaxs, names = readtxt(file)
            # dealpath = path + "/" + filename + ".xml"
            dealpath = xmlPath + "/" + filename + ".xml"
            filename = filename + '.jpg'
            with open(dealpath, 'w') as f:
                writexml(dealpath, filename, num, xmins, ymins, xmaxs, ymaxs, names)


# 读取图片的高和宽写入xml
def dealwh(path):
    files = os.listdir(path)  # 列出所有文件
    for file in files:
        filename = os.path.splitext(file)[0]  # 分割出文件名
        sufix = os.path.splitext(file)[1]  # 分割出后缀
        if sufix == '.jpg':
            height, width = readsize(file)
            # dealpath = path + "/" + filename + ".xml"
            dealpath = xmlPath + "/" + filename + ".xml"
            gxml(dealpath, height, width)


# 在xml文件中添加宽和高
def gxml(path, height, width):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    heights = root.getElementsByTagName('height')[0]
    heights.firstChild.data = height
    # print(height)

    widths = root.getElementsByTagName('width')[0]
    widths.firstChild.data = width
    # print(width)
    with open(path, 'w') as f:
        # with open(xmlPath, 'w') as f:
        dom.writexml(f)
    return


# 创建xml文件
def writexml(path, filename, num, xmins, ymins, xmaxs, ymaxs, names, height='256', width='256'):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "VOC2007"

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = "%s" % filename

    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(num):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '%s' % names[i]
        node_name = SubElement(node_object, 'pose')
        node_name.text = '%s' % "unspecified"
        node_name = SubElement(node_object, 'truncated')
        node_name.text = '%s' % "0"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % xmins[i]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % ymins[i]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % xmaxs[i]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % ymaxs[i]

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    with open(path, 'wb') as f:
        f.write(xml)
    return


def readsize(p):
    p_file = imagePath + "/" + p
    img = Image.open(p_file)
    width = img.size[0]
    height = img.size[1]
    return height, width


if __name__ == "__main__":
    # path = ("D:/NWPU VHR-10 dataset/NWPU VHR-10 dataset/test")
    imagePath = ("./WeedData/JPEGImages")
    jsonPath = ("./WeedData/original_json")
    xmlPath = ("./WeedData/Annotations")
    deal(jsonPath)
    dealwh(imagePath)
