import torch 
from dataset import OfflineDataset
from torch.utils.data import DataLoader
import argparse
import os
from utils import load_config, seed_everything
from env import make_env
import agents as Agents
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Q Learning")
    
    parser.add_argument("-config", type=str, default="config1.yaml",
                        help="Set config file")
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = cfg["gpu"]
    
    seed_everything(seed=cfg["seed"])
    env = make_env(cfg["env_name"])
    
    print("create dataset")
    train_dataset = OfflineDataset(data_dir=cfg["data_dir"], log_file=cfg["log_file"])
    print("number of data: ", len(train_dataset))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg["batch_size"],
                                  shuffle=True,
                                  num_workers=os.cpu_count())
    print("get agent")
    agent_ = getattr(Agents, cfg["algo"])
    agent = agent_(gamma=cfg["gamma"], epsilon=cfg["epsilon"], lr=cfg["lr"],
                     eps=cfg["eps"],
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=cfg["mem_size"], eps_min=cfg["eps_min"],
                     batch_size=cfg["batch_size"], replace=cfg["replace"], eps_dec=cfg["eps_dec"],
                     chkpt_dir=cfg["chkpt_dir"], algo=cfg["algo"],
                     env_name=cfg["env_name"], n_norm_groups=cfg["n_norm_groups"],
                     dataloader=train_dataloader, alpha=cfg["alpha"],
                     v_min=cfg["v_min"], v_max=cfg["v_max"], atoms=cfg["atoms"])
    
    print("training")
    best_score = -np.inf
    scores = []
    for i in range(cfg["epochs"]):
        print("learn")
        train_loss = agent.learn()
        print(f"Epoch: {i+1} | "
              f"train_loss: {train_loss:.4f} |")
        
        
        
        
    
    
