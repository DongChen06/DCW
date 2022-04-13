import cv2


def draw_boxes(image, bboxes, format='yolo'):
    """
    Function accepts an image and bboxes list and returns
    the image with bounding boxes drawn on it.

    Parameters
    :param image: Image, type NumPy array.
    :param bboxes: Bounding box in Python list format.
    :param format: One of 'coco', 'voc', 'yolo' depending on which final
        bounding noxes are formated.

    Return
    image: Image with bounding boxes drawn on it.
    box_areas: list containing the areas of bounding boxes.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    box_areas = []

    if format == 'coco':
        # coco has bboxes in xmin, ymin, width, height format
        # we need to add xmin and width to get xmax and...
        # ... ymin and height to get ymax
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0])+int(box[2])
            ymax = int(box[1])+int(box[3])
            width = int(box[2])
            height = int(box[3])
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=4
            )
            box_areas.append(width*height)

    if format == 'voc':
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            width = xmax - xmin
            height = ymax - ymin
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width*height) 

    if format == 'yolo':
        # need the image height and width to denormalize...
        # ... the bounding box coordinates
        h, w, _ = image.shape
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for box_num, box in enumerate(bboxes):
            if int(box[4]) == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            x1, y1, x2, y2 = yolo2bbox(box)
            # denormalize the coordinates
            xmin = int(x1*w)
            ymin = int(y1*h)
            xmax = int(x2*w)
            ymax = int(y2*h)
            width = xmax - xmin
            height = ymax - ymin
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=color,
                thickness=24
            ) 
            box_areas.append(width*height) 
    return image, box_areas


def yolo2bbox(bboxes):
    """
    Function to convert bounding boxes in YOLO format to 
    xmin, ymin, xmax, ymax.
    
    Parmaeters:
    :param bboxes: Normalized [x_center, y_center, width, height] list

    return: Normalized xmin, ymin, xmax, ymax
    """
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax