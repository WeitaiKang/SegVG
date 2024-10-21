export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export OMP_NUM_THREADS=4

# please change the dataset name, eval_set and the eval_model path
# the following is an example of evaluating on the gref_umd test dataset
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10030 --use_env eval.py \
--batch_size 32 --num_workers 4 \
--bert_enc_num 12 --detr_enc_num 6 --backbone resnet101 --dataset gref_umd --max_query_len 40 \
--eval_set test \
--data_root /data_ext1/kangweitai/VG/ \
--split_root /data_ext1/kangweitai/VG/split/ \
--eval_model /data_ext1/kangweitai/VG/output/SegVG/gref_umd/best_checkpoint.pth