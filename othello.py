import numpy as np
import cv2
import PIL.Image as Image
import gym
import random
from gym import Env, spaces
import os

font = cv2.FONT_HERSHEY_TRIPLEX


mouse_X = -1
mouse_Y = -1
click_count = 0

EMP, BLK, WHT = 0, -1, 1
GRID_STATE = {"Black":BLK, "White":WHT, "Empty":EMP}

class Piece:
    def __init__(self, color):
        self.color = color
        assert color in ["Black", "White"], "invalid color"
        self.shape = (50, 50)  


class Black(Piece):
    def __init__(self):
        super(Black, self).__init__(color="Black")
        self.icon = cv2.imread(os.path.join("images", "black.png"))


class White(Piece):
    def __init__(self):
        super(White, self).__init__(color="White")
        self.icon = cv2.imread(os.path.join("images", "white.png"))


class Othello:
    def __init__(self, human_VS_machine=False):
        super(Othello, self).__init__()
        self.canvas_shape = (620, 520, 3)  
        self.observation_shape = (8, 8)  
        self.observation_space = spaces.Discrete(8 * 8) 

        self.action_space = spaces.Discrete(8 * 8)  
        self.board = cv2.imread(os.path.join("images", "board.png"))  
       
        self.grid_coordinates = (
            ((131, 32), (131, 90), (131, 148), (131, 206), (131, 264), (131, 323), (131, 381), (131, 439)),
            ((191, 32), (191, 90), (191, 148), (191, 206), (191, 264), (191, 323), (191, 381), (191, 439)),
            ((249, 32), (249, 90), (249, 148), (249, 206), (249, 264), (249, 323), (249, 381), (249, 439)),
            ((308, 32), (308, 90), (308, 148), (308, 206), (308, 264), (308, 323), (308, 381), (308, 439)),
            ((366, 32), (366, 90), (366, 148), (366, 206), (366, 264), (366, 323), (366, 381), (366, 439)),
            ((425, 32), (425, 90), (425, 148), (425, 206), (425, 264), (425, 323), (425, 381), (425, 439)),
            ((483, 32), (483, 90), (483, 148), (483, 206), (483, 264), (483, 323), (483, 381), (483, 439)),
            ((541, 32), (541, 90), (541, 148), (541, 206), (541, 264), (541, 323), (541, 381), (541, 439)))
        self.black = Black() 
        self.white = White()
        self.black_counter_coor = (135, 75)  
        self.white_counter_coor = (258, 75)  
        self.winner_display_coor = (392, 75)  
        self.prompt_display_coor = (136, 34) 
        self.next_player = "Black"  
        self.black_count = 2  
        self.white_count = 2  
        self.done = False
        self.next_possible_actions = set() 
        self.show_next_possible_actions_hint = True  
        self.dirs = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)) 
        self.human_VS_machine = human_VS_machine

    def reset(self):

        self.grids = [[EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, BLK, WHT, EMP, EMP, EMP],
                      [EMP, EMP, EMP, WHT, BLK, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      ]  
        self.black_count = 2 
        self.white_count = 2  
        self.next_player = "Black"  
        self.done = False
        self.next_possible_actions = self.__get_possible_actions(color="Black")
        self.__refresh_canvas()
        self.__img_counter = 0  
        info = {"next_player": self.next_player, "next_possible_actions": self.next_possible_actions}
        return self.grids, info

    def __refresh_canvas(self):
        self.canvas = self.board.copy()
       
        for i in range(8):
            for j in range(8):
                if self.grids[i][j] == GRID_STATE["Empty"]: continue
                r, c = self.grid_coordinates[i][j]
                if self.grids[i][j] == GRID_STATE["Black"]: self.canvas[r:r + self.black.shape[0], c:c + self.black.shape[1]] = self.black.icon
                else: self.canvas[r:r + self.white.shape[0], c:c + self.white.shape[1]] = self.white.icon
        self.canvas = cv2.putText(self.canvas, str(self.black_count), self.black_counter_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, str(self.white_count), self.white_counter_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)

        if self.show_next_possible_actions_hint:
            for i, j in self.next_possible_actions:
                coor = (self.grid_coordinates[i][j][1]+15, self.grid_coordinates[i][j][0]+32)
                self.canvas = cv2.putText(self.canvas, "+", coor, font, 0.6, (188, 199, 137), 1, cv2.LINE_AA)

        if self.done:
            self.canvas = cv2.putText(self.canvas, "Game Over", self.prompt_display_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)
            if self.black_count > self.white_count:
                self.canvas = cv2.putText(self.canvas, "Black", self.winner_display_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)
            elif self.black_count < self.white_count:
                self.canvas = cv2.putText(self.canvas, "White", self.winner_display_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)
            else:
                self.canvas = cv2.putText(self.canvas, "Tie", self.winner_display_coor, font, 0.8, (188, 199, 137), 2, cv2.LINE_AA)
        else:
            self.canvas = cv2.putText(self.canvas, self.next_player, self.prompt_display_coor, font, 0.7, (188, 199, 137), 1, cv2.LINE_AA)

    def render(self, mode='human'):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            
            cv2.imshow("Othello", self.canvas)
           
           
            cv2.waitKey(200)
        elif mode == "rgb_array":
            return self.canvas  

    def step(self, action: tuple):

        assert len(action) == 2, "Invalid Action"
        assert self.action_space.contains(action[0]*8+action[1]), "Invalid Action"
        assert action in self.next_possible_actions, "Invalid Action"
        done = False
        color_self = GRID_STATE[self.next_player]
        color_opponent = GRID_STATE["White"] if self.next_player=="Black" else GRID_STATE["Black"]
        reward = 0.

        self.__put(action[0], action[1], self.next_player)
        

        next_player = "Black" if self.next_player == "White" else "White" 
        next_possible_actions = self.__get_possible_actions(next_player)
        if not next_possible_actions:  
            next_player = "Black" if self.next_player == "Black" else "White"  
            next_possible_actions = self.__get_possible_actions(next_player)
            if not next_possible_actions:
                self.next_possible_actions = set()
                self.next_player = None
                done = True  
            else:
                reward += 0  
                self.next_player = next_player
                self.next_possible_actions = next_possible_actions
        else:
            self.next_player = next_player  
            self.next_possible_actions = next_possible_actions

        
        self.done = done
        info = {"next_player": self.next_player, "next_possible_actions": self.next_possible_actions}
        if done:
            conclusion = "Gave Over! "
            if self.black_count == self.white_count:  
                reward += 2
                info["winner"] = "Tie"
                conclusion += "No winner, ends up a Tie"
            elif self.black_count > self.white_count:
                info["winner"] = "Black"
                reward += 10 if color_self == GRID_STATE["Black"] else -10
                conclusion += "Winner is Black."
            else:
                info["winner"] = "White"
                reward += 10 if color_self == GRID_STATE["White"] else -10
                conclusion += "Winner is White."
            if self.human_VS_machine:
                print(conclusion)
        self.__refresh_canvas()
        return self.grids, reward, done, info

    def __put(self, i:int, j:int, color:str):

        assert self.grids[i][j] == GRID_STATE["Empty"], "Cannot put a piece in a occupied grid"
        assert color in ("Black", "White"), "illegal color input"
        color_self = GRID_STATE[color]
        color_opponent = GRID_STATE["White"] if color == "Black" else GRID_STATE["Black"]

        self.grids[i][j] = color_self
       
        flips = []

        def check_flip(dir, dist, opponent_cnt, i, j, candidates, flips):

            if i < 0 or i >= 8 or j < 0 or j >= 8: return  
            if self.grids[i][j] == GRID_STATE["Empty"]: return  
            if self.grids[i][j] == color_self: 
                if opponent_cnt == 0: return  
                if opponent_cnt > 0:  
                    flips += candidates
                    return
            if self.grids[i][j] == color_opponent:  
                check_flip(dir, dist+1, opponent_cnt+1, i+dir[0], j+dir[1], candidates+[(i, j)], flips)

        for dir in self.dirs:
            check_flip(dir=dir, dist=1, opponent_cnt=0, i=i+dir[0], j=j+dir[1], candidates=[], flips=flips)

        flips = set(flips)
        for f_i, f_j in flips: 
            self.grids[f_i][f_j] = color_self
        if color == "Black":
            self.white_count, self.black_count = self.white_count - len(flips), self.black_count + len(flips) + 1
        else:
            self.white_count, self.black_count = self.white_count + len(flips) + 1, self.black_count - len(flips)

    def __get_possible_actions(self, color):

        assert color in ("Black", "White"), "illegal color input"
        actions = []
        color_self = GRID_STATE[color]
        color_opponent = GRID_STATE["White"] if color == "Black" else GRID_STATE["Black"]

        def check_possible(dir, dist, opponent_cnt, i, j, actions):

            if i < 0 or i >= 8 or j < 0 or j >= 8: return  
            if dist == 1 and self.grids[i][j] == GRID_STATE["Empty"]: return  
            if self.grids[i][j] == color_self: return  
            if self.grids[i][j] == GRID_STATE["Empty"] and opponent_cnt>0:  
                actions.append((i, j))
                return
            if self.grids[i][j] == color_opponent: 
                check_possible(dir, dist+1, opponent_cnt+1, i+dir[0], j+dir[1], actions)

        for r in range(8):
            for c in range(8):
                if self.grids[r][c] in (GRID_STATE["Empty"], color_opponent): continue
                for dir in self.dirs:
                    check_possible(dir=dir, dist=1, opponent_cnt=0, i=r+dir[0], j=c+dir[1], actions=actions)
        return set(actions)  

    def get_random_action(self):
        if self.next_possible_actions:
            return random.choice(list(self.next_possible_actions))
        return ()

    def close(self):
        cv2.destroyAllWindows()

    def __piece_count(self):

        self.black_count, self.white_count = 0, 0
        for row in self.grids:
            for p in row:
                if p < 0: continue
                self.black_count += 1 if p==0 else 0
                self.white_count += 1 if p==1 else 0

    def __check_termination(self):

        if self.next_player is None:
            return True
        if sum([sum([p < 0 for p in row]) for row in self.grids]) == 0:  
            return True
        return False

    def get_human_action(self):
        global mouse_X, mouse_Y, click_count
        mouse_X, mouse_Y, click_count = -1, -1, 0

        def get_mouse_location(event, x, y, flags, param):
            global mouse_X, mouse_Y, click_count
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_X, mouse_Y = x, y
                click_count += 1

        while click_count < 3:  
            cv2.waitKey(20)
            cv2.setMouseCallback("Othello", get_mouse_location)
            if mouse_X < 0 or mouse_Y < 0: continue  
            for i in range(8):
                for j in range(8):
                    y1, x1 = self.grid_coordinates[i][j]
                    y2, x2 = y1 + 49, x1 + 49
                    if y1 <= mouse_Y <= y2 and x1 <= mouse_X <= x2 and (i, j) in self.next_possible_actions:
                        return i, j
            print("illegal action")
            mouse_X, mouse_Y = -1, -1
        return self.get_random_action()


