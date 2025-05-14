EXPERIMENT_NAME="rtdetrv2_r50vd_organoid_stomach_cancer_epoch50"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node=2 main/test_track_EIoU.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --ckpt output/exp_train/exp_train_rtdetrv2_r50vd_organoid_stomach_cancer_epoch50/best.pth \
  --output_dir output/exp_track/${EXPERIMENT_NAME}