export GNN_TYPE=egnn
export dataset_k=10

CUDA_VISIBLE_DEVICES=3 python pretrain.py \
    --date 0824 \
    --batch_token_num 20480 \
    --lambda1 0.05 \
    --lambda2 0.5 \
    --c_alpha_max_neighbors $dataset_k \
    --use_sasa \
    --p 0.5 \
    --gnn $GNN_TYPE \
    --layer_num 6 \
    --gnn_config src/Egnnconfig/$GNN_TYPE.yaml \
    --protein_dataset data/cath40_k$dataset_k \
    --mutant_dataset data/proteingym_valid