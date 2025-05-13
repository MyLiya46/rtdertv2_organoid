EXPERIMENT_NAME="rtdetrv2_r50vd_organoid_stomach_cancer_epoch50_freeze_at_2"

CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29500 --nproc_per_node=1 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --resume output/exp_train/rtdetrv2_r50vd_organoid_stomach_cancer_epoch50_freeze_at_4/best.pth \
  --output-dir output/exp_train/${EXPERIMENT_NAME} \
  --use-amp \
  --seed 0
