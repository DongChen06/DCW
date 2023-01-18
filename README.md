# DeepWeed Benchmark: Benchmarking the Deep Learning Techniques for Weed Recognition
Weeds are among major threats to crop production including cotton. Traditional weed control that relies on broadcast, 
repeated application of herbicides, is facing challenges with managing herbicide-resistance of weeds while reducing impacts on environments. 
Machine vision (MV)-based weed control offers a promising weeding solution based on the recognition of weeds in images followed by localized, 
precision treatments for weed removal. Artificial intelligence (AI) through deep learning (DL) is emerging as a 
key driver to the development of MV technology for weed detection and control. To realize the potential of AI for weed control 
requires a systematic evaluation of DL models with large-scaled, ground-truthed weed datasets for weed detection. 
In this study, we present a novel benchmark DeepCottonWeeds (DCW), of DL techniques for weed detection tasks in cotton production systems. 
The DCW is extensible modular, and unified; it standardizes the process of weed recognition tasks by: 1) developing a scalable and diverse dataset, 2) modularizing DL implementations, and 3) unifying the evaluation protocol. 
A comprehensive benchmark of state-of-art deep learning algorithms for weed detection will be established. By leveraging the DCW pipeline, end users can readily focus on the development of robust deep learning models with automated data processing and experimental evaluations. 
Datasets and source codes will be made publicly available. 


## 1. Installation
- Create a conda environment: `conda create -n cottonweeddetection python=3.8 -y`
- Active the virtul environment: `conda activate cottonweeddetection`
- Install requirements: `pip install -r requirements.txt`

## 2. Preparing the Dataset
### 2.1 Dataset Preparation
- Download the data (https://doi.org/10.5281/zenodo.7535814) and unzip to the *datasets/* folder.
- Run the script to convert the labeled data into YOLO-V5 format: `python commons/vig2yolov5.py`
- (Optional) Run the script to convert the labeled data into COCO format: `python commons/yolov52coco.py`
- The generated dataset should have the following format:
- Partition the dataset to different folder: `python commons/pationing_dataset_yolov5.py --outputDir [USE ABOSULTE ADDRESS]/DCW/datasets/Dataset_final`, for example:
`python commons/pationing_dataset_yolov5.py --outputDir /home/dong9/PycharmProjects/DCW/datasets/Dataset_final`

You can also download the preprocessed dataset directly at: [here](xx) and put it to the *datasets* folder.

### 2.2 (Optional) Data Augmentation
- To augment the trianing images, one can refer to `Data_augmentation/data_augmentation_v1.py`
- Sample images are also provided in `datasets/Data_aug/`, one can use `Data_augmentation/data_augmentation.py` to generate examples.

### 2.3 (Optiona2) Dataset Analysis
- To analysis the dataset, we can run: `python commons/dataset_analysis.py`.
- To analysis the dataset focusing on the Top-12 classes in a given folder, run: `python commons/dataset_analysis_top12.py --imageDir datasets/Dataset_final/DATA_0/val`.

### 2.4 (Optiona3) Prepare your Own Dataset 
- Label the dataset using [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/).
- Then transfer the dataset using the above two steps to convert the dataset to YOLOv5 or COCO formats.

## 3. Training and Testing
- Download the pretrained models from [here](https://drive.google.com/drive/folders/1s_72kdEM6N2J9uklgH8P30HnAcjFkJ1X?usp=sharing)
and unzip to corresponding folders. For example, you need to put the `yolov3.pt`, `yolov3-spp.pt` and `yolov3-tiny.pt` under the *YOLOV3/* folder.
- We trained the models for 5 replications on 5 GPUs, readers are recommended to look at the `train_cudax.sh` files. For instance, to run the 0st data folder, we can run:
`bash -i train_cuda0.sh`.
- To test the models, we can run: `bash -i test0.sh`.

## 4. Performance
The YOLO algorithms[1-6] used for our experiments are not maintained by us, please give credit to the authors of the YOLO algorithms[1-6].

# Video Demos
The video demos can be accessed at [[Video Demos]](https://drive.google.com/drive/folders/1Xvb-KvzDlX5IfAtGbEwZaSIkTvij25rZ?usp=share_link)

# Citation
If you find the models and or the dataset useful, consider citing the follow article:

Dang, F., Chen, D., Lu, Y., Li, Z., 2023. YOLOWeeds: A novel benchmark of YOLO object detectors for multi-class weed detection in cotton production systems. Computers and Electronics in Agriculture. https://doi.org/10.1016/j.compag.2023.107655.


# Reference
- [1-1] YOLOv3: Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
- [1-2] YOLOv3 Implementation: https://github.com/ultralytics/yolov3.
- [2-1] YOLOv4: Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).
- [2-2] YOLOv4 Implementation: https://github.com/WongKinYiu/PyTorch_YOLOv4.
- [3-1] YOLOv5: None
- [3-2] YOLOv5 Implementation: https://github.com/ultralytics/yolov5.
- [4-1] YOLOR: Wang, Chien-Yao, I-Hau Yeh, and Hong-Yuan Mark Liao. "You Only Learn One Representation: Unified Network for Multiple Tasks." arXiv preprint arXiv:2105.04206 (2021).
- [4-2] YOLOR Implementation: https://github.com/WongKinYiu/yolor.
- [5-1] YOLOX: Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).
- [5-2] YOLOX Implementation: https://github.com/Megvii-BaseDetection/YOLOX.
- [6-1] ScaledYOLOv4: Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "Scaled-yolov4: Scaling cross stage partial network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
- [6-2] ScaledYOLOv4 Implementation: https://github.com/WongKinYiu/ScaledYOLOv4.
