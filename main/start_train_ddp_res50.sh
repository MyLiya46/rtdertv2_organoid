CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29500 --nproc_per_node=1 main/train.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --resume output/exp_train_rtdetrv2_r50vd_organoid_stomach_cancer_epoch50/best.pth \
  --output-dir output/rtdetrv2_r50vd_organoid \
  --use-amp \
  --seed 0
