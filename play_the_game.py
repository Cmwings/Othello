import cv2
import torch
import os
import matplotlib.pyplot as plt
from othello import Othello
from bot import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    HUMAN_vs_MACHINE = False   # Change to True for Human vs Machine
    env = Othello(human_VS_machine=HUMAN_vs_MACHINE)
    
    # Load the best trained model for the machine player
    machine = DQN("White", device=device, model_name="Best_Model").to(device)
    machine.load_model(machine.model_name)
    
    win_count = 0
    loss_count = 0
    tie_count = 0
    
    if HUMAN_vs_MACHINE: 
        # Human vs. Machine game play
        for ep in range(1):
            obs, info = env.reset()  
            env.render()  
            while True:
                if info["next_player"] == "White":  
                    action = machine.choose_action(obs, info["next_possible_actions"])
                else:  
                    action = env.get_human_action()
                obs, _, done, info = env.step(action)
                env.render()
                if done:
                    break
            cv2.waitKey(3000)  # Wait for 3 seconds before closing
        cv2.waitKey()
        env.close()
    else:
        # Machine vs. Machine game play
        results = []  
        round_max = 500
        RENDER = False
        for ep in range(round_max):
            obs, info = env.reset()  
            while True:
                if info["next_player"] == "White":  
                    action = machine.choose_action(obs, info["next_possible_actions"])  
                else:
                    action = env.get_random_action()  
                obs, _, done, info = env.step(action)
                if done:
                    if info["winner"] == "White":
                        win_count += 1
                        results.append(1) 
                    elif info["winner"] == "Black":
                        loss_count += 1
                        results.append(-1)  
                    else:
                        tie_count += 1
                        results.append(0)  
                    print("Round: {:d}/{:d}, winner is ".format(ep + 1, round_max), info["winner"])
                    break  
        
        total_games = win_count + loss_count + tie_count
        win_percentage = (win_count / total_games) * 100
        loss_percentage = (loss_count / total_games) * 100
        tie_percentage = (tie_count / total_games) * 100

        # Plotting win, loss, tie counts with percentages
        categories = ['Wins', 'Losses', 'Ties']
        counts = [win_count, loss_count, tie_count]
        percentages = [win_percentage, loss_percentage, tie_percentage]

        plt.bar(categories, counts, color=['green', 'red', 'blue'])
        for i, count in enumerate(counts):
            plt.text(i, count, f'{percentages[i]:.2f}%', ha='center', va='bottom')

        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.title('Win-Loss-Tie Counts with Percentages')
        plt.show()

        env.close()
