# Training script on CUDA 0 and for seed 0
# bash -i train_cuda0.sh

# YOLOv3
cd ~/PycharmProjects/DCW/YOLOv3; /home/dong9/anaconda3/envs/cottonweeddetection/bin/python train.py --img 640 --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov3-tiny.pt
cd ~/PycharmProjects/DCW/YOLOv3; /home/dong9/anaconda3/envs/cottonweeddetection/bin/python train.py --img 640 --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov3.pt
cd ~/PycharmProjects/DCW/YOLOv3; /home/dong9/anaconda3/envs/cottonweeddetection/bin/python train.py --img 640 --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov3-spp.pt


# YOLOv4
cd ~/PycharmProjects/DCW/YOLOv4;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --device 0 --batch-size 16 --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolov4.cfg --weights 'yolov4.pt' --name yolov4
cd ~/PycharmProjects/DCW/YOLOv4;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --device 0 --batch-size 16 --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolov4-csp-leaky.cfg --weights 'yolov4-csp-leaky.pt' --name yolov4-pacsp-leaky
cd ~/PycharmProjects/DCW/YOLOv4; /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --device 0 --batch-size 16 --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolov4-csp-s-leaky.cfg --weights 'yolov4-csp-s-leaky.pt' --name yolov4-pacsp-s-leaky
cd ~/PycharmProjects/DCW/YOLOv4; /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --device 0 --batch-size 16 --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolov4-csp-x-leaky.cfg --weights 'yolov4-csp-x-leaky.pt' --name yolov4-pacsp-x-leaky


# YOLOv5
cd ~/PycharmProjects/DCW/YOLOv5;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --img 640  --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov5n.pt
cd ~/PycharmProjects/DCW/YOLOv5;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --img 640  --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov5s.pt
cd ~/PycharmProjects/DCW/YOLOv5;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --img 640  --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov5m.pt
cd ~/PycharmProjects/DCW/YOLOv5;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --img 640  --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov5l.pt
cd ~/PycharmProjects/DCW/YOLOv5;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --img 640  --device 0 --batch 16 --epochs 5 --data cottonweedsdetection_seed0.yaml --weights yolov5x.pt


# YOLOR
cd ~/PycharmProjects/DCW/YOLOR;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --batch-size 16 --device 0  --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolor_csp.cfg --weights 'yolor_csp.pt' --name yolor_csp --hyp hyp.scratch.640.yaml
cd ~/PycharmProjects/DCW/YOLOR;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --batch-size 16 --device 0  --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolor_csp_x.cfg --weights 'yolor_csp_x.pt' --name yolor_csp_x --hyp hyp.scratch.640.yaml
cd ~/PycharmProjects/DCW/YOLOR;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --batch-size 16 --device 0  --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolor_p6.cfg --weights 'yolor_p6.pt' --name yolor_p6 --hyp hyp.scratch.640.yaml
cd ~/PycharmProjects/DCW/YOLOR;  /home/dong9/anaconda3/envs/cottonweeddetection/bin/python  train.py --batch-size 16 --device 0  --img 640 640 --epochs 5 --data cottonweedsdetection_seed0.yaml --cfg cfg/yolor_w6.cfg --weights 'yolor-w6.pt' --name yolor_w6 --hyp hyp.scratch.640.yaml

#