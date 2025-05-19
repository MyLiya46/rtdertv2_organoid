python main/test_track_EIoU.py \
  --config configs/rtdetrv2/rtdetrv2_r50vd_organoid_labeled_all_linux.yml \
  --ckpt output/exp_train/rtdetrv2_r50vd_organoid_labeled_all_epoch50_freeze_at_0_another50epoch/best.pth \
  --output_dir output/exp_track/rtdetrv2_r50vd_organoid_labeled_all_freeze3stage
