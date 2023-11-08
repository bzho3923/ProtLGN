export GNN_TYPE=egnn
export dataset_k=10

CUDA_VISIBLE_DEVICES=6 python loc_pretrain.py \
    --ckpt_dir result/loc_ \
    --max_train_epochs 100 \
    --lr 1e-4 \
    --max_nodes 4096 \
    --gnn $GNN_TYPE \
    --layer_num 6 \
    --gnn_config src/Egnnconfig/$GNN_TYPE.yaml