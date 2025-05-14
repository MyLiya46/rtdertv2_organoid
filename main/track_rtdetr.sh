EXPERIMENT_NAME="rtdetrv2_r50vd_organoid_stomach_cancer_epoch50_freeze"

torchrun --nproc_per_node=2 main/track_rtdetr.py \
    -f your_exp.py \
    -c your_ckpt.pth.tar \
    --output_dir output/exp_track/${EXPERIMENT_NAME} \
