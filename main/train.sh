EXPERIMENT_NAME="lr1e-4_bs8_amp"

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29500 --nproc_per_node=2 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r101vd_6x_organoid_linux.yml \
  --tuning checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth \
  --output-dir output/exp_train/${EXPERIMENT_NAME} \
  --use-amp \
  --seed 0
