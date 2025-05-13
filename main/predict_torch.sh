CUDA_VISIBLE_DEVICES= 0, 1 torchrun --master_port=29500 --nproc_per_node=2 main/predict_torch.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_linux.yml \
  --ckpt output/exp_train_rtdetrv2_r50vd_organoid_stomach_cancer_epoch50/best.pth \
  --output_dir output/exp_predict
