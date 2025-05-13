CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29500 --nproc_per_node=1 main/test_track_EIoU.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --ckpt output/exp_train_rtdetrv2_r50vd_organoid_stomach_cancer_epoch50/best.pth