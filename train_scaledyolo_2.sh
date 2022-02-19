# Training script on CUDA 3, 5
# bash -i train_scaledyolo_2.sh

# ScaledYOLOv4
cd /localscratch2/zhengyu/Documents/DCW/ScaledYOLOv4; /localscratch/zhengyu/.conda/envs/cottonweeddetection/bin/python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 16 --img 640 640 --data cottonweedsdetection_seed1.yaml --cfg yolov4-p5.yaml --weights 'yolov4-p5.pt' --sync-bn --device 3,5 --name yolov4-p5
cd /localscratch2/zhengyu/Documents/DCW/ScaledYOLOv4; /localscratch/zhengyu/.conda/envs/cottonweeddetection/bin/python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 16 --img 640 640 --data cottonweedsdetection_seed1.yaml --cfg yolov4-p6.yaml --weights 'yolov4-p6.pt' --sync-bn --device 3,5 --name yolov4-p6
cd /localscratch2/zhengyu/Documents/DCW/ScaledYOLOv4; /localscratch/zhengyu/.conda/envs/cottonweeddetection/bin/python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 16 --img 640 640 --data cottonweedsdetection_seed1.yaml --cfg yolov4-p7.yaml --weights 'yolov4-p7.pt' --sync-bn --device 3,5 --name yolov4-p7
