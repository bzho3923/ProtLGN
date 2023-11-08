import argparse
import warnings
import torch
import os, time
import sys
import yaml
import json
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader
from numpy import nan
from torch_geometric.data import Batch
from scipy.stats import spearmanr
from transformers import logging
from accelerate import Accelerator


##    Loading-model-params:plot,learning-rate and Batch-Sampler.
from src.utils.draw_utils import plot_model
from src.utils.train_utils import lr_scheduler
from src.utils.data_utils import BatchSampler
from src.utils.utils import print_param_num


##    Loading-model and Data
from model import GNN_model
from data import get_dataset

# set path
current_dir=os.getcwd()
sys.path.append(current_dir)
#ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def epoch_runner(args, loader, stage="train"):
    ### $\lambda1$ and $\lambda2$ in paper.
    lambda1=torch.tensor(args.lambda1)
    lambda2=torch.tensor(args.lambda2)
    model.train()
    total_loss = 0
    total_loss_list = [0,0,0,0,0,0]
    epoch_bar = tqdm(loader)
    for data_list in epoch_bar:  # Iterate in batches over the training dataset.
        batch_graph = Batch.from_data_list(data_list).cuda()
        out = model(batch_graph)
        loss1 = criterion(out[:,:20], batch_graph.y)
        total_loss_list[1] += loss1.item()
        loss = loss1
        dimention = 20
        if args.use_sasa:
            loss2 = criterion2(out[:, dimention], batch_graph.y1)
            total_loss_list[2] += loss2.item()
            loss = loss + lambda1 * loss2
            dimention = dimention + 1
        if args.use_bfactor:
            loss3 = criterion2(out[:, dimention], batch_graph.y2)
            total_loss_list[3] += loss3.item()
            loss = loss + lambda1 * loss3
            dimention = dimention + 1
        if args.use_dihedral:
            loss4 = criterion2(out[:, dimention], batch_graph.y3)
            loss5 = criterion2(out[:, dimention + 1], batch_graph.y4)
            loss6 = criterion2(out[:, dimention + 2], batch_graph.y5)
            loss7 = criterion2(out[:, dimention + 3], batch_graph.y6)
            loss8 = criterion2(out[:, dimention + 4], batch_graph.y7)
            loss9 = criterion2(out[:, dimention + 5], batch_graph.y8)
            total_loss_list[4] += (
                loss4.item()+loss5.item()+loss6.item()+loss7.item()+loss8.item()+loss9.item()
            )
            loss = loss + lambda2 * (loss4 + loss5 + loss6 + loss7 + loss8 + loss9)
            dimention = dimention + 6
        if args.use_coordinate:
            loss10 = criterion2(out[:, dimention : dimention + 3], batch_graph.y9)
            total_loss_list[5] += loss10.item()
            loss = loss + lambda1 * (loss10)
            dimention = dimention + 3
        
        if stage == "train":
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        total_loss_list[0] += loss.item()
        epoch_bar.set_postfix(loss=round(loss.item(), 2))
    return total_loss / len(loader), total_loss_list


def mutant_predict(model, loader, protein_names):
    model.eval()
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    spear_cor = np.zeros(protein_num)
    spear_cor_multi = [[] for _ in range(protein_num)]
    row_contents = [[] for _ in range(protein_num)]
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

            spear_cor[protein_idx] = spearmanr(
                df_score["true_score"], df_score["pred_score"]
            ).correlation
            if spear_cor[protein_idx] is nan:
                spear_cor[protein_idx] = 0
            spear_cor_multi[protein_idx] = np.zeros(
                (len(df_score["mutat_pt_num"].unique()), 3)
            )
            count = 0
            spear_cor_only_multi = spearmanr(
                df_score[df_score["mutat_pt_num"] != 1]["true_score"],
                df_score[df_score["mutat_pt_num"] != 1]["pred_score"],
            ).correlation

            for mutat_num, group in df_score.groupby("mutat_pt_num"):
                spear_cor_multi[protein_idx][count, 0] = mutat_num
                spear_cor_multi[protein_idx][count, 1] = spearmanr(
                    group["true_score"], group["pred_score"]
                ).correlation
                spear_cor_multi[protein_idx][count, 2] = len(group["true_score"])
                count += 1

            row_contents[protein_idx] = [
                protein_names[protein_idx],
                int(spear_cor_multi[protein_idx][:, 2].sum()),
                spear_cor[protein_idx],
                int(spear_cor_multi[protein_idx][1:, 2].sum()),
                spear_cor_only_multi,
            ]
            for j in range(spear_cor_multi[protein_idx].shape[0]):
                row_contents[protein_idx].append(
                    int(spear_cor_multi[protein_idx][j, 2])
                )
                if spear_cor_multi[protein_idx][j, 1] is nan:
                    row_contents[protein_idx].append("nan")
                else:
                    row_contents[protein_idx].append(spear_cor_multi[protein_idx][j, 1])
            # print(protein_names[protein_idx], spear_cor[protein_idx])
    print("-"*40)
    all_mutat_num = sum(row_contents[i][1] for i in range(protein_num))
    spear_info = {}
    w_avg = 0
    for i in range(protein_num):
        w_avg += spear_cor[i] * row_contents[i][1] / all_mutat_num
        spear_info[protein_names[i]] = spear_cor[i]
    
    print(f"multi_avg: {spear_cor.mean()}")
    print(f"multi_w_avg: {w_avg}")
    
    return spear_info, spear_cor.mean(), w_avg


def create_parser():
    parser = argparse.ArgumentParser()

    # train strategy
    parser.add_argument("--p",type=float,default=0.5,
                        help="please select the noise probability of labelnoise")
    parser.add_argument("--use_sasa",action="store_true",
                        help="whether to use the sasa feature")
    parser.add_argument("--use_bfactor",action="store_true",
                        help="whether to use the bfactor feature")
    parser.add_argument("--use_dihedral",action="store_true",
                        help="whether to use the dihedral feature")
    parser.add_argument("--use_coordinate",action="store_true",
                        help="whether to use the coordinate feature")
    parser.add_argument("--lambda1",type=float,default=0.2,
                        help="lambda1 in sasa,bfactor,corrdinate loss")
    parser.add_argument("--lambda2",type=float,default=0.5,
                        help="lambda2 in dihedral loss")
    parser.add_argument("--use_denoise",action="store_true",
                        help="whether to ues denoise")
    parser.add_argument("--noise_type",type=str,default="wild",
                        help="what kind of noise adding on protein, either wild or substitute")
    parser.add_argument("--date",type=str,default="Sep_25th",
                        help="date using save the filename")
    parser.add_argument("--gnn",type=str,default="egnn",
                        help="GNN gin, gin-virtual, or gcn, or gcn-virtual or egnn (default: gin-virtual)")

    # train model
    parser.add_argument("--problem_type",type=str,default="aa_classification")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--init_lr",type=float,default=1e-7,
                        help="init learning rate for warmup")
    parser.add_argument("--warmup",type=int,default=0,
                        help="warm up step")
    parser.add_argument("--weight_decay",type=float,default=1e-2,
                        help="weight_decay")
    parser.add_argument("--num_classes",type=int,default=20,
                        help="number of GNN output (default: 20)")
    parser.add_argument("--epochs",type=int,default=300,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--step_schedule",type=int,default=350,
                        help="number of epoch schedule lr")
    parser.add_argument("--batch_token_num",type=int,default=4096,
                        help="how many tokens in one batch")
    parser.add_argument("--max_graph_token_num",type=int,default=4000,
                        help="max token num a graph has")
    parser.add_argument("--node_dim",type=int,default=26,
                        help="number of node feature")
    parser.add_argument("--edge_dim",type=int,default=93,
                        help="number of edge feature")
    parser.add_argument("--layer_num",type=int,default=6,
                        help="number of layer")
    parser.add_argument("--dropout",type=float,default=0,
                        help="dropout rate")
    parser.add_argument("--subtitude_label",action="store_true",
                        help="whether smooth the label by subtitude table")
    parser.add_argument("--JK",type=str,default="last",
                        help="using what nodes embedding to make prediction,last or sum")
    parser.add_argument("--portion",type=int,default=40,
                        help="mix ratio of af and cath dataset")
    parser.add_argument("--clip",type=float,default=4.0,
                        help="mix ratio of af and cath dataset")
    
    # dataset 
    parser.add_argument("--mix_dataset",action="store_true",
                        help="whether mix alphafold and cath dataset")
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


if __name__ == "__main__":
    args=create_parser()

    #config model and cuda.
    gnn_config=yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]


    # load dataset and split dataset.
    train_dataset, val_dataset, test_dataset, mutant_dataset = get_dataset(args)
    def collect_fn(batch):
        return batch
    cath_dataloader = lambda dataset: DataLoader(
        dataset=dataset, num_workers=32, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            dataset, 
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=True
            )
        )
    train_loader, val_loader, test_loader = map(
        cath_dataloader, (train_dataset, val_dataset, test_dataset)
        )
    protein_names = mutant_dataset.get_names()

    mutant_loader = pygDataLoader(mutant_dataset, batch_size=1, shuffle=False)

    # Define file-name and model config details
    filename=(
            args.date
            + f'_K={args.c_alpha_max_neighbors}_p={args.p}_sasa={args.use_sasa}_'
              f'bfactor={args.use_bfactor}_'
              f'lambda1={args.lambda1}_lambda2={args.lambda2}_'
              f'noise={args.noise_type}_'
              f'gnn={args.gnn}_layer={args.layer_num}_drop={args.dropout}_lr={args.lr}'
    )
    print(filename)
    model = GNN_model(gnn_config, args)
    model.cuda()
    print_param_num(model)

    accelerator = Accelerator()
    # Define-Loss function and optimizer.
    criterion=torch.nn.CrossEntropyLoss()
    criterion2=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epochs // 2, gamma=0.1
    )

    train_loss_list, valid_loss_list = [], []
    mutant_mean_list = []
    loss_sum, loss_cla, loss_sas, loss_bfa, loss_dih, loss_cor = [] ,[] ,[] ,[] ,[] ,[]
    
    best_record = -100000

    #Training.
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, loss_list = epoch_runner(args, train_loader, "train")
        train_loss_list.append(train_loss)
        scheduler.step()
        loss_sum.append(loss_list[0] / len(train_loader))
        loss_cla.append(loss_list[1] / len(train_loader))
        loss_sas.append(args.lambda1 * loss_list[2] / len(train_loader))
        loss_bfa.append(args.lambda1 * loss_list[3] / len(train_loader))
        loss_dih.append(args.lambda2 * loss_list[4] / len(train_loader))
        loss_cor.append(args.lambda1 * loss_list[5] / len(train_loader))

        with torch.no_grad():
            valid_loss, valid_loss_list = epoch_runner(args, val_loader, "valid")

        spear_info, multi_mean, multi_weight = mutant_predict(
            model, mutant_loader, protein_names
        )
        with open("result/spear_info.json", "w") as f:
            json.dump(spear_info, f)

        mutant_mean_list.append(multi_mean)
        # single correlation achieve best performance

        if multi_mean > best_record:
            best_record = multi_mean
            torch.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_list": train_loss_list,
                    "valid_loss_list": valid_loss_list,
                    "mutation_mean": multi_mean,
                    "spear_info": spear_info
                },
                "result/weight/" + filename + "coeff.pt",
            )
        # multi correlation achieve best performance
        
        print(
            f"Epoch:{epoch:03d}, Train loss:{train_loss:.4f}, Valid loss:{valid_loss:.4f},take {time.time()-start:.2f} s"
        )
        plot_model(
            epoch, train_loss_list, loss_cla, loss_sas, loss_bfa, loss_dih,
            loss_cor, valid_loss_list, mutant_mean_list, filename,
        )
