import argparse, os
import numpy as np
from dqn_agent import DQNAgent
import agents as Agents
from gym import wrappers
from env import make_env
from utils import load_config, seed_everything
import csv
import collections
import datetime


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
    env = make_env(cfg["env_name"], clip_rewards=False, no_ops=30, fire_first=False)
    print("observation_shape; ", env.observation_space.shape)
    best_score = -np.inf
    agent_ = getattr(Agents, cfg["algo"])
    agent = agent_(gamma=cfg["gamma"], epsilon=cfg["epsilon"], lr=cfg["lr"],
                     eps=cfg["eps"],
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=cfg["mem_size"], eps_min=cfg["eps_min"],
                     batch_size=cfg["batch_size"], replace=cfg["replace"], eps_dec=cfg["eps_dec"],
                     chkpt_dir=cfg["chkpt_dir"], algo=cfg["algo"],
                     env_name=cfg["env_name"], n_norm_groups=cfg["n_norm_groups"],
                     dataloader=None, alpha=cfg["alpha"],
                     v_min=cfg["v_min"], v_max=cfg["v_max"], atoms=cfg["atoms"])
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = timestamp + "_" + agent.algo + "_" + agent.env_name + "_lr" + str(agent.lr) + "_" + \
            "_" + str(cfg["n_games"]) + "games"
    
    if cfg["save_memory"]:
        memory_dir = f"{cfg['memory_dir']}/{fname}"
        os.mkdir(memory_dir)
    
    if cfg["load_checkpoint"]:
        agent.load_models()
        
    if cfg["make_video"]:
        video_dir = f"{cfg['video_dir']}/{fname}"
        os.mkdir(video_dir)
        env = wrappers.Monitor(env, video_dir, 
                            video_callable=lambda x: x%1==0,
                            force=True)
        
    n_steps = 0
    n_save_data = 0
    scores = collections.deque(maxlen=100)
    
    for i in range(cfg["n_games"]):
        done = False
        score = 0
        observation, _  = env.reset()
        steps_per_episode = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            if steps_per_episode >= 10000:
                done = True
            score += reward
            if not cfg["load_checkpoint"]:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
            if cfg["save_memory"] and n_steps % 100 == 0:
                transition = [observation, action, reward, observation_, int(done)]
                np.save(f"{memory_dir}/{n_save_data}.npy", np.array(transition))
                n_save_data += 1
 
            observation = observation_
            n_steps += 1
            steps_per_episode += 1
        scores.append(score)
        
        avg_score = np.mean(scores)
        print("episode ", i, "score: ", score,
              "average score %.1f best score %.1f epsilon %.2f" %
              (avg_score, best_score, agent.epsilon),
              "steps ", n_steps, "n_save_data ", n_save_data)
        
        if avg_score > best_score:
            if not cfg["load_checkpoint"]:
                agent.save_models()
            best_score = avg_score
        
        if i == 0:
            logfname = f"{cfg['log_dir']}/{fname}.csv"
            with open(logfname, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "score", "average_score", "best_score", "epsilon", "steps", "n_save_data"])
        else:
            with open(logfname, "a") as f:
                writer = csv.writer(f)
                writer.writerow([i, score, avg_score, best_score, agent.epsilon, n_steps, n_save_data])