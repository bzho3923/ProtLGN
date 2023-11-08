import argparse
import torch
import os, time
import yaml
import shutil
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader
from numpy import nan
from scipy.stats import spearmanr

from src.utils.utils import print_param_num
from model import GNN_model
from data import mutant_dataset

def mutant_predict(model, loader, protein_names):
    model.eval()
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    spear_cor = np.zeros(protein_num)
    mutant_bar = tqdm(loader)
    with torch.no_grad():
        for data in mutant_bar:
            ### calculate in model
            graph_data = data.cuda()
            out = model(graph_data)
            out = torch.log(softmax(out[:, :20]) + 1e-9)

            ## find protein name
            protein_idx = data.protein_idx
            score_info = data.score_info[0]
            num_mutat = len(score_info)
            true_score = torch.zeros(num_mutat)
            pred_score = torch.zeros(num_mutat)
            mutat_pt_num = torch.zeros(num_mutat, dtype=torch.int64)

            # prepare dataframe
            for mutat_idx in range(num_mutat):
                mutat_info, true_score[mutat_idx] = score_info[mutat_idx]
                mutat_pt_num[mutat_idx] = len(mutat_info)
                for i in range(mutat_pt_num[mutat_idx]):
                    item = mutat_info[i]
                    if int(item[1]) > out.shape[0]:
                        continue
                    pred_score[mutat_idx] += (
                        out[int(item[1] - 1), int(item[2])]
                        - out[int(item[1] - 1), int(item[0])]
                    ).cpu()
            df_score = {
                "true_score": true_score.reshape(-1).numpy(), 
                "pred_score": pred_score.reshape(-1).numpy(), 
                "mutat_pt_num": mutat_pt_num.reshape(-1).numpy()
                }
            df_score = pd.DataFrame(df_score)
            df_score.to_csv(f"result/{args.mutant_dataset.split('/')[-1]}/{protein_names[protein_idx]}.tsv", sep="\t", index=False)
            spear_cor[protein_idx] = spearmanr(
                df_score["true_score"], df_score["pred_score"]
            ).correlation
            if spear_cor[protein_idx] is nan:
                spear_cor[protein_idx] = 0
            
    print("-"*40)
    spear_info = {}
    for i in range(protein_num):
        spear_info[protein_names[i]] = spear_cor[i]
    
    print(f"multi_avg: {spear_cor.mean()}")
    
    return spear_info, spear_cor.mean()


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_sasa",action="store_true",
                        help="whether to use the sasa feature")
    parser.add_argument("--use_bfactor",action="store_true",
                        help="whether to use the bfactor feature")
    parser.add_argument("--use_dihedral",action="store_true",
                        help="whether to use the dihedral feature")
    parser.add_argument("--use_coordinate",action="store_true",
                        help="whether to use the coordinate feature")
    parser.add_argument("--gnn",type=str,default="egnn",
                        help="GNN gin, gin-virtual, or gcn, or gcn-virtual or egnn (default: gin-virtual)")

    # train model
    parser.add_argument("--problem_type",type=str,default="aa_classification")
    parser.add_argument("--num_classes",type=int,default=20,
                        help="number of GNN output (default: 20)")
    parser.add_argument("--node_dim",type=int,default=26,
                        help="number of node feature")
    parser.add_argument("--edge_dim",type=int,default=93,
                        help="number of edge feature")
    parser.add_argument("--layer_num",type=int,default=6,
                        help="number of layer")
    parser.add_argument("--dropout",type=float,default=0,
                        help="dropout rate")
    parser.add_argument("--checkpoint",type=str,default="",
                        help="which model used to load")
    
    # dataset 
    parser.add_argument("--c_alpha_max_neighbors",type=int,default=10,
                        help="the parameter of KNN which used construct the graph, 10 or 20")

    #Attention: If you have dataset,you can change these with your dataset!
    parser.add_argument("--protein_dataset",type=str,default="data/cath40_k10_dyn_imem",
                        help="main protein dataset")
    parser.add_argument("--mutant_dataset",type=str,default="data/evaluation",
                        help="mutation dataset")
    parser.add_argument("--gnn_config",type=str,default="src/Egnnconfig/egnn.yaml",
                        help="gnn config")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=create_parser()

    #config model and cuda.
    gnn_config=yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]

    # load dataset and split dataset.
    mutant = mutant_dataset(args)
    
    protein_names = mutant.get_names()

    mutant_loader = pygDataLoader(mutant, batch_size=1, shuffle=False)

    model = GNN_model(gnn_config, args)
    model.cuda()
    print_param_num(model)
    model_state_dict = torch.load(args.checkpoint)["model_state_dict"]
    # new_state_dict = {}
    # for key, value in model_state_dict.items():
    #     new_key = "GNN_model." + key
    #     new_state_dict[new_key] = value
    model.load_state_dict(model_state_dict)
    
    os.makedirs(f"result/{args.mutant_dataset.split('/')[-1]}", exist_ok=True)
    # for protein in protein_names:
    #     shutil.copyfile(f"{args.mutant_dataset}/{protein}/{protein}.tsv", f"result/{args.mutant_dataset.split('/')[-1]}/{protein}.tsv")
    
    spear_info, multi_mean = mutant_predict(
        model, mutant_loader, protein_names
    )