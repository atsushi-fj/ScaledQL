import random
import os
import numpy as np
import torch
import datetime
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_curve(data_path):
    df = pd.read_csv(data_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(df["episode"], df["epsilon"], color="C0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y", color="C0")
    ax.set_title("DQN")
    
    ax2.plot(df["episode"], df["average_score"], color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Average score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")
    plt.show()


def create_boxplot(cql_path, sql_path, c51_sql_path):
    cql_df = pd.read_csv(cql_path)
    sql_df = pd.read_csv(sql_path)
    c51_sql_df = pd.read_csv(c51_sql_path)
    cql_df["model"] = "CQL"
    sql_df["model"] = "Scaled QL \n(ResNet18 / MSE)"
    c51_sql_df["model"] = "Scaled QL \n(ResNet18 / C51)"
    df = pd.concat([cql_df, sql_df, c51_sql_df], axis=0)
    fig = plt.figure()
    sns.boxplot(x="model", y="score", data=df)
    plt.tight_layout()
    plt.savefig("./plots/eval_2.png")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
    
    
def create_display_name(experiment_name,
                        model_name,
                        extra=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

    if extra:
        name = f"{timestamp}-{experiment_name}-{model_name}-{extra}"
    else:
        name = f"{timestamp}-{experiment_name}-{model_name}"
    print(f"[INFO] Create wandb saving to {name}")
    return name


def load_config(file="config1.yaml"):
    """Load config file"""
    config_path = Path("./config/")
    with open(config_path / file, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg
