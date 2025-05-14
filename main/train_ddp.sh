EXPERIMENT_NAME="test_of_ddp_training"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node=2 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --resume output/exp_train/rtdetrv2_r50vd_organoid_stomach_cancer_epoch50_freeze_at_4/best.pth \
  --output_dir output/exp_train/${EXPERIMENT_NAME} \
  --use-amp \
  --seed 0
