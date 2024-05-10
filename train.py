import cv2
import torch
import os
import matplotlib.pyplot as plt
from othello import Othello
from bot import DQN
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    white_checkpoint = "White_Model"  
    black_checkpoint = "Black_Model"  
    
    # Initialize White and Black agents
    agent_White = DQN("White", device=device).to(device)
    agent_Black = DQN("Black", device=device).to(device)
    
    # Load pre-trained models if available
    if os.path.exists(white_checkpoint):
        agent_White.load_model(white_checkpoint)
    if os.path.exists(black_checkpoint):
        agent_Black.load_model(black_checkpoint)

    env = Othello(human_VS_machine=False)
    winning_rate = []
    best_model, best_winning_rate = None, 0.
    is_White = []
    max_epoch = 10000
    dominant_counter_white = 0
    RENDER = False
    
    # Training loop
    for ep in range(1, max_epoch + 1):
        ep_reward = []
        obs, info = env.reset()
        done = False
        if RENDER:
            env.render()
        while True:
            next_player = info["next_player"]
            next_possible_actions = info["next_possible_actions"]

            if next_player == "White":
                action = agent_White.choose_action(obs, next_possible_actions)
                obs_, reward, done, info = env.step(action)
                ep_reward.append(reward)
                agent_White.store_transition(obs, action, reward, done, obs_, next_possible_actions)
            else:
                action = agent_Black.choose_action(obs, next_possible_actions)

                obs_, reward, done, info = env.step(action)
                if done:
                    if info["winner"] == "Black":
                        agent_White.reward_transition_update(-10.)
                    elif info["winner"] == "White":
                        agent_White.reward_transition_update(10.)
                    else:
                        agent_White.reward_transition_update(1.)
            obs = copy.deepcopy(obs_)

            if RENDER:
                env.render()
            if done:
                loss = agent_White.learn()
                print("ep: {:d}/{:d}, white player loss value: {:.4f}".format(ep, max_epoch, loss))
                is_White.append(True if info["winner"] == "White" else False)
                break
        
        # Update winning rate and check for model dominance
        if ep % 20 == 0:
            winning_rate.append(np.mean(is_White))
            is_White = []

            if best_winning_rate <= winning_rate[-1]:
                best_model = copy.deepcopy(agent_White)
                best_winning_rate = winning_rate[-1]
            if winning_rate[-1] >= 0.60:
                dominant_counter_white += 1
            else:
                dominant_counter_white = 0
            if dominant_counter_white >= 3:
                dominant_counter_white = 0
                agent_Black.weights_assign(agent_White.brain_evl)
              
    # Save trained models
    agent_White.save_model("White_Model")
    agent_Black.save_model("Black_Model")
    best_model.save_model("Best_Model")

    # Plot winning rate
    plt.plot(range(20, max_epoch + 1, 20), winning_rate, color='green') 
    plt.ylabel('Win Ratio')
    plt.xlabel('Episode')
    plt.title('Winning Rate Over Episodes')
    plt.show()
