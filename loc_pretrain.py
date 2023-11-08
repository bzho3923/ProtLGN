import argparse
import warnings
import torch
import os
import sys
import yaml
import wandb
import random
from torch_geometric.data import Data
from torch import nn
from tqdm import tqdm
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader as pygDataLoader
from torch_geometric.data import Batch
from transformers import logging
from torchmetrics.classification import BinaryAccuracy
from accelerate import Accelerator
from time import strftime, localtime
from src.utils.data_utils import BatchSampler
from src.utils.utils import print_param_num
##    Loading-model and Data
from model import GNN_model

# set path
current_dir=os.getcwd()
sys.path.append(current_dir)
#ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def train(args, model, accelerator, train_loader, val_loader, test_loader, optimizer, device):
    best_acc = 0
    path = os.path.join(args.ckpt_dir, args.model_name)
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        model.train()
        train_loss, train_acc = loop(model, accelerator, train_loader, epoch, optimizer, device)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f} acc: {train_acc:.4f}')
        
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(model, accelerator, val_loader, epoch, device=device)
            wandb.log({"valid/val_loss": val_loss, "valid/val_acc": val_acc, "valid/epoch": epoch})
        print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path)
            print(f'>>> BEST at epcoh {epoch}, acc: {best_acc:.4f}')
            print(f'>>> Save model to {path}')
        
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        loss, acc = loop(model, accelerator, test_loader, device=device)
        wandb.log({"test/test_loss": loss, "test/test_acc": acc})
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')

    
def loop(model, accelerator, dataloader, epoch=0, optimizer=None, device=None):
    total_loss, total_acc = 0, 0
    iter_num = len(dataloader)
    global_steps = epoch * len(dataloader)
    epoch_iterator = tqdm(dataloader)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    # labels = []
    # x_means = []
    # x_input = []
    for batch in epoch_iterator:
        batch.to(device)
        label = torch.as_tensor(batch.label, dtype=torch.float32).to(device)
        label = label.argmax(-1)
        logits, x_mean = model(batch)
        input_x = scatter_mean(batch.x, batch.batch, dim=0)
        # x_input.append(input_x)
        # labels.append(label)
        # x_means.append(x_mean)
        # print(input_x.shape, label.shape, x_mean.shape)
        # print(logits.squeeze(-1).cpu(), label.cpu())
        loss = loss_fn(logits.squeeze(-1), label)
        total_loss += loss.item()
        preds = logits.argmax(-1)
        
        acc = int(torch.sum(preds == label).cpu()) / len(label)
        total_acc += acc
        
        global_steps += 1
        
        if optimizer:
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_postfix(train_loss=loss.item(), train_acc=acc)
            wandb.log({"train/train_loss": loss.item(), "train/train_acc": acc, "train/epoch": epoch}, step=global_steps)
        else:
            epoch_iterator.set_postfix(eval_loss=loss.item(), eval_acc=acc)
    
    # labels = torch.cat(labels, dim=0)
    # x_means = torch.cat(x_means, dim=0)
    # x_input = torch.cat(x_input, dim=0)
    # torch.save({"labels": labels, "out": x_means, "input": x_input}, "labels_x.pt")
    
    epoch_loss = total_loss / iter_num
    epoch_acc = total_acc / iter_num
    return epoch_loss, epoch_acc



def create_parser():
    parser = argparse.ArgumentParser()

    # train strategy
    parser.add_argument("--gnn",type=str,default="egnn",
                        help="GNN gin, gin-virtual, or gcn, or gcn-virtual or egnn (default: gin-virtual)")
    parser.add_argument("--use_sasa",action="store_true",
                        help="whether to use the sasa feature")
    parser.add_argument("--use_bfactor",action="store_true",
                        help="whether to use the bfactor feature")
    parser.add_argument("--use_dihedral",action="store_true",
                        help="whether to use the dihedral feature")
    parser.add_argument("--use_coordinate",action="store_true",
                        help="whether to use the coordinate feature")
    # train model
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    parser.add_argument('--model_name', default=None, help='model name')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--problem_type",type=str,default="multi_label_classification")
    parser.add_argument("--num_labels",type=int,default=10, help="number of localizations")
    parser.add_argument('--max_nodes', type=int, default=3000, help='max number of nodes per batch')
    parser.add_argument('--max_train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument("--node_dim",type=int,default=26, help="number of node feature")
    parser.add_argument("--edge_dim",type=int,default=93, help="number of edge feature")
    parser.add_argument("--layer_num",type=int,default=6, help="number of layer")
    parser.add_argument("--dropout",type=float,default=0, help="dropout rate")
    parser.add_argument("--clip",type=float,default=4.0, help="mix ratio of af and cath dataset")
    
    # dataset 
    parser.add_argument("--c_alpha_max_neighbors",type=int,default=10,
                        help="the parameter of KNN which used construct the graph, 10 or 20")

    #Attention: If you have dataset,you can change these with your dataset!
    parser.add_argument("--gnn_config",type=str,default="src/Egnnconfig/egnn.yaml",
                        help="gnn config")
    # wandb log
    parser.add_argument('--wandb_project', type=str, default='LGN_loc_debug')
    parser.add_argument("--wandb_entity", type=str, default="matwings")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    
    if args.wandb_run_name is None:
        args.wandb_run_name = f"LGN_lr{args.lr}"
    if args.model_name is None:
        args.model_name = f"{args.wandb_run_name}.pt"
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    wandb.init(
        project=args.wandb_project, name=args.wandb_run_name, 
        entity=args.wandb_entity, config=vars(args)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gnn_config=yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]

    train_dataset = torch.load("data/location/Locationk10/processed/train.pt")
    val_dataset = torch.load("data/location/Locationk10/processed/val.pt")
    test_dataset = torch.load("data/location/Locationk10/processed/test.pt")
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 3):
        print(">>> ", train_dataset[i])
    
    
    loc_dataloader = lambda dataset: pygDataLoader(
        dataset=dataset, 
        num_workers=4, 
        batch_sampler=BatchSampler(
            dataset,
            [d.num_nodes for d in dataset],
            batch_token_num=args.max_nodes
            )
        )
    train_loader, val_loader, test_loader = map(
        loc_dataloader, (train_dataset, val_dataset, test_dataset)
    )

    model = GNN_model(gnn_config, args)
    model.to(device)
    print_param_num(model)
    # model_state_dict = torch.load(args.load_ckpt)["model_state_dict"]
    # new_state_dict = {}
    # for key, value in model_state_dict.items():
    #     new_key = "GNN_model." + key
    #     new_state_dict[new_key] = value
    # model.load_state_dict(new_state_dict)

    accelerator = Accelerator()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    print("---------- Start Training ----------")
    train(args, model, accelerator, train_loader, val_loader, test_loader, optimizer, device=device)
    wandb.finish()