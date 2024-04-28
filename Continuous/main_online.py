import argparse
import gym
import numpy as np
import csv
import datetime

from agents import SACAgent
from utils import load_config
from env import ImageEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Soft Actor Critic")
    
    parser.add_argument("-config", type=str, default="config1.yaml",
                        help="Set config file")
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    
    env = gym.make(cfg["env_id"])
    env = ImageEnv(env)
    input_dims = (cfg["n_channels"], cfg["img_size"], cfg["img_size"])
    agent = SACAgent(input_dims=input_dims, action_dim=env.action_space.shape[0], auto_alpha=cfg["auto_alpha"], alpha=cfg["alpha"],
                     chkpt_dir=cfg["chkpt_dir"], lr_actor=cfg["lr_actor"], lr_critic=cfg["lr_critic"], 
                     gamma=cfg["gamma"], hidden_size=cfg["hidden_size"], tau=cfg["tau"], memory_size=cfg["memory_size"],
                     batch_size=cfg["batch_size"])
        
    dateinfo = datetime.datetime.now()
    logfname = f"{cfg['log_dir']}/{cfg['experiment_name']}_{dateinfo.strftime('%Y%m%d%H%M%S')}.csv"
    with open(logfname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "avg_score", "n_save_data"])

    if cfg["load_checkpoint"]:
        agent.load_models()
    episode_reward_list = []
    eval_reward_list = []
    best_score = env.reward_range[0]
    n_steps = 0
    n_update = 0
    running_score = 0
    n_save_data = 0
    for i_episode in range(1, cfg["n_episode"] + 1):

        score = 0
        done = False
        truncated = False
        state, _ = env.reset()

        while True:
            
            if cfg["start_steps"] > n_steps:
                action = env.action_space.sample()
            else:
                
                action = agent.select_action(state)

            if len(agent.memory) > cfg['batch_size']:
                agent.update_parameters()
                n_update += 1

            next_state, reward, terminated, truncated, done, _ = env.step(action)
            n_steps += 1
            score += reward

            if terminated or done or truncated:
                mask = float(not True)
                agent.memory.push(state=state, action=action, reward=reward, next_state=next_state, mask=mask)
            else:
                mask = float(not False)
                agent.memory.push(state=state, action=action, reward=reward, next_state=next_state, mask=mask)
                
            if cfg["save_memory"] and n_steps % cfg["save_interval"] == 0:
                transition = [state, action, reward, next_state, mask]
                np.save(f"{cfg['memory_dir']}/{n_save_data}.npy", np.array(transition, dtype=object))
                n_save_data += 1

            state = next_state

            if terminated or done or truncated:
                break

        running_score = running_score * 0.99 + score * 0.01
        if running_score > best_score:
                best_score = running_score
                if not cfg["load_checkpoint"]:
                    agent.save_models()
        episode_reward_list.append(score)
        print("episode", i_episode, "score %.1f" % score,
                "average score %.1f" % running_score)
        
        with open(logfname, "a") as f:
            writer = csv.writer(f)
            writer.writerow([i_episode, score, running_score, n_save_data])

        
                
    
