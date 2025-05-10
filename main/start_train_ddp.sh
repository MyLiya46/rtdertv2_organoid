CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --master_port=29500 --nproc_per_node=2 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r101vd_6x_organoid_linux.yml \
  --tuning checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
  --use-amp \
  --seed 0 \
  --output-dir output/exp_rtdetrv2_r101vd_organoid \
  > log.txt 2>&1 &
