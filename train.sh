export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export OMP_NUM_THREADS=8

# please change your output_dir and tensorboard according to the dataset name
# if you want to resume training, please set the resume path

# for unc dataset
# detr_model = detr-r101-unc.pth
# dataset = unc

# for unc+ dataset
# detr_model = detr-r101-unc.pth
# dataset = unc+

# for gref dataset
# detr_model = detr-r101-gref.pth
# dataset = gref

# for gref_umd dataset
# detr_model = detr-r101-gref.pth
# dataset = gref_umd

# for referit dataset
# detr_model = detr-r101-referit.pth
# dataset = referit

# The following is an example of training on the unc dataset
python -m torch.distributed.launch --nproc_per_node=8 --master_port=10033 --use_env train.py \
--batch_size 8 --aug_scale --aug_translate --aug_crop --backbone resnet101 \
--detr_model /data_ext1/kangweitai/VG/backbone/detr-r101-unc.pth \
--lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --num_workers 16 \
--dataset unc --max_query_len 40 --freeze_epochs 10 --lr_drop 60 --epochs 1 --clip_max_norm 0.1 \
--data_root /data_ext1/kangweitai/VG/ \
--split_root /data_ext1/kangweitai/VG/split/ \
--output_dir /data_ext1/kangweitai/VG/output/SegVG/unc/ \
--tensorboard /data_ext1/kangweitai/VG/tensorboard/SegVG/unc/ \
# --resume /data_ext1/kangweitai/VG/output/SegVG/unc/checkpoint_latest.pth