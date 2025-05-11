CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29500 --nproc_per_node=1 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r101vd_6x_organoid.yml \
  --tuning checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
  --seed 0 \
  --use-amp
