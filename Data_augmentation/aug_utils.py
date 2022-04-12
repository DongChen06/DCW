import albumentations as A


def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = max(0., min(float(x[0]), 1.))
            x[1] = max(0., min(float(x[1]), 1.))
            x[2] = max(0., min(float(x[2]), 1.))
            x[3] = max(0., min(float(x[3]), 1.))
            coords.append(x)
    return coords


def writeYolo(coords, out_dir):
    with open(out_dir, "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))


def getTransform(loop):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
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
            A.JpegCompression(quality_lower=0, quality_upper=1, p=0.2)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 6:
        transform = A.Compose([
            A.FancyPCA(alpha=0.2,  p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 7:
        transform = A.Compose([
            A.Blur( p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 8:
        transform = A.Compose([
            A.GaussNoise(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 9:
        transform = A.Compose([
            A.RGBShift(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))

    return transform