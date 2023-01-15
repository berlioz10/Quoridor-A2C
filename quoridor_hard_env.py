from collections import deque
import queue
from time import sleep
import gym
from gym import spaces
import numpy as np

PAWN_MOVES = 4
WALL_PLACES = 128


class QuoridorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(QuoridorEnv, self).__init__()
        # first 4 moves will be the pawn moves ( 0 - up 1 - down 2 - right 3 - left)
        # the other ones will be considered the wall places in this order:
        # wall_place_X = (value - 4) / 2 / 8
        # wall_place_Y = (value - 4) / 2 % 8
        # wall_place_position = value % 2, for horizontal, 1 for vertical
        self.action_space = spaces.Discrete(PAWN_MOVES + WALL_PLACES)
        self.reward = 0
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(69,), dtype=np.float64)

    def print_matrix(self):
        print_matrix = [[' ' for _ in range(17)] for _ in range(17)]
        for i in range(len(self.wallPlaces)):
            for j in range(len(self.wallPlaces[0])):
                if self.wallPlaces[i][j] == 1:
                    print_matrix[i * 2 + 1][j * 2 + 1] = '#'
                    print_matrix[i * 2 + 1][j * 2 + 0] = '#'
                    print_matrix[i * 2 + 1][j * 2 + 2] = '#'
                elif self.wallPlaces[i][j] == 2:
                    print_matrix[i * 2 + 1][j * 2 + 1] = '#'
                    print_matrix[i * 2 + 0][j * 2 + 1] = '#'
                    print_matrix[i * 2 + 2][j * 2 + 1] = '#'

        for i in range(9):
            for j in range(9):
                print_matrix[i * 2][j * 2] = '-'
        
        print_matrix[self.positionLeePlayer[0] * 2][self.positionLeePlayer[1] * 2] = 'O'
        print_matrix[self.positionAIPlayer[0] * 2][self.positionAIPlayer[1] * 2] = 'X'

        for row in print_matrix:
            for el in row:
                print(el, end=" ")
            print()
        print("\n\n")

    def find_path(self, wallPlaces, visited: list, i, j, correct):
        if [i, j] in visited:
            return False

        if i == correct:
            return True
        
        if i < 0 or i > 8 or j < 0 or j > 8:
            return False

        visited.append([i, j])

        ok_left = False
        ok_right = False
        ok_up = False
        ok_down = False

        if i > 0:
            if not (j == 0 and wallPlaces[i - 1][0] == 1) and not (j == 8 and wallPlaces[i - 1][7] == 1):
                if j != 0 and j != 8:         
                    if not (wallPlaces[i - 1][j] == 1 or wallPlaces[i - 1][j - 1] == 1):
                        ok_up = self.find_path(wallPlaces, visited, i - 1, j, correct)
                else:
                    ok_up = self.find_path(wallPlaces, visited, i - 1, j, correct)

        if i < 8:
            if not (j == 0 and wallPlaces[i][0] == 1) and not (j == 8 and wallPlaces[i][7] == 1):
                if j != 0 and j != 8:
                    if not (wallPlaces[i][j] == 1 or wallPlaces[i][j - 1] == 1):
                        ok_down = self.find_path(wallPlaces, visited, i + 1, j, correct)
                else:
                    ok_down = self.find_path(wallPlaces, visited, i + 1, j, correct)

        if j > 0:
            if not (i == 0 and wallPlaces[0][j - 1] == 2) and not (i == 8 and wallPlaces[7][j - 1] == 2):
                if i != 0 and i != 8:
                    if not (wallPlaces[i][j - 1] == 2 or wallPlaces[i - 1][j - 1] == 2):
                        ok_left = self.find_path(wallPlaces, visited, i, j - 1, correct)
                else:
                    ok_left = self.find_path(wallPlaces, visited, i, j - 1, correct)
        if j < 8:
            if not (i == 0 and wallPlaces[0][j] == 2) and not (i == 8 and wallPlaces[7][j] == 2):
                if i != 0 and i != 8:
                    if not (wallPlaces[i][j] == 2 or wallPlaces[i - 1][j] == 2):
                        ok_right = self.find_path(wallPlaces, visited, i, j + 1, correct)
                else:
                    ok_right = self.find_path(wallPlaces, visited, i, j + 1, correct)


        return ok_up or ok_down or ok_right or ok_left

    def check_end(self):
        if self.positionAIPlayer[0] == 0:
            return 1
        if self.positionLeePlayer[0] == 8:
            return 2
        return 0

    def enemys_move(self):
        # implement lees algorithm
        okRemakeLee = False
        if self.distance_lee == -1:
            okRemakeLee = True
        for i in range(len(self.wallPlaces)):
            for j in range(len(self.wallPlaces[0])):
                if self.wallPlaces[i][j] == 1:
                    if self.lee_matrix[i * 2 + 1][j * 2] == -2 or self.lee_matrix[i * 2 + 1][j * 2 + 1] == -2 or self.lee_matrix[i * 2 + 1][j * 2 + 2] == -2:
                        okRemakeLee = True
                    self.lee_matrix[i * 2 + 1][j * 2 + 1] = -1
                    self.lee_matrix[i * 2 + 1][j * 2 + 0] = -1
                    self.lee_matrix[i * 2 + 1][j * 2 + 2] = -1
                elif self.wallPlaces[i][j] == 2:
                    if self.lee_matrix[i * 2][j * 2 + 1] == -2 or self.lee_matrix[i * 2 + 1][j * 2 + 1] == -2 or self.lee_matrix[i * 2 + 2][j * 2 + 1] == -2:
                        okRemakeLee = True
                    self.lee_matrix[i * 2 + 1][j * 2 + 1] = -1
                    self.lee_matrix[i * 2 + 0][j * 2 + 1] = -1
                    self.lee_matrix[i * 2 + 2][j * 2 + 1] = -1
        
        for i, row in enumerate(self.lee_matrix):
            for j, el in enumerate(row):
                if el != -1 and el != -2:
                    self.lee_matrix[i][j] = 0
                if okRemakeLee == True:
                    if el == -2:
                        self.lee_matrix[i][j] = 0
        print(okRemakeLee)
        if okRemakeLee == True:
            q = []
            q.append([self.positionLeePlayer[0] * 2, self.positionLeePlayer[1] * 2])
            self.lee_matrix[self.positionLeePlayer[0] * 2][self.positionLeePlayer[1] * 2] = 1
            while len(q) != 0:
                [i, j] = q.pop(0)
                if i > 0:
                    if (self.lee_matrix[i - 1][j] == 0 and self.lee_matrix[i - 2][j] == 0) or (self.lee_matrix[i - 2][j] > self.lee_matrix[i][j] + 1 and self.lee_matrix[i - 1][j] != -1):
                        self.lee_matrix[i - 2][j] = self.lee_matrix[i][j] + 1
                        q.append([i - 2, j])

                if i < 16:    
                    if (self.lee_matrix[i + 1][j] == 0 and self.lee_matrix[i + 2][j] == 0) or (self.lee_matrix[i + 2][j] > self.lee_matrix[i][j] + 1 and self.lee_matrix[i + 1][j] != -1):
                        self.lee_matrix[i + 2][j] = self.lee_matrix[i][j] + 1
                        q.append([i + 2, j])

                if j > 0:
                    if (self.lee_matrix[i][j - 1] == 0 and self.lee_matrix[i][j - 2] == 0)  or (self.lee_matrix[i][j - 2] > self.lee_matrix[i][j] + 1 and self.lee_matrix[i][j - 1] != -1):
                        self.lee_matrix[i][j - 2] = self.lee_matrix[i][j] + 1
                        q.append([i, j - 2])

                if j < 16:
                    if (self.lee_matrix[i][j + 1] == 0 and self.lee_matrix[i][j + 2] == 0)  or (self.lee_matrix[i][j + 2] > self.lee_matrix[i][j] + 1 and self.lee_matrix[i][j + 1] != -1):
                        self.lee_matrix[i][j + 2] = self.lee_matrix[i][j] + 1
                        q.append([i, j + 2])
            min = 82 # :D
            min_pos = -1
            for i in range(16):
                if min > self.lee_matrix[16][i] and self.lee_matrix[16][i] != -1 and self.lee_matrix[16][i] != 0:
                    min = self.lee_matrix[16][i]
                    min_pos = i

            if min_pos == -1:
                print("Oh no")
                for el in self.lee_matrix:
                    print(el)
                print("Oh no")
            else:
                if self.distance_lee != -1:
                    if self.distance_lee < min:
                        self.reward += 50    
                self.distance_lee = min - 1
                ct = min
                i = 16
                j = min_pos
                self.lee_matrix[i][j] = -2
                while ct != 2:
                    ok = False
                    if j > 0 and ok != True:
                        if self.lee_matrix[i][j - 2] == ct - 1:
                            self.lee_matrix[i][j - 1] = -2
                            ct -= 1
                            j  -= 2
                            ok = True
                    if j < 16 and ok != True:
                        if self.lee_matrix[i][j + 2] == ct - 1:
                            self.lee_matrix[i][j + 1] = -2
                            ct -= 1
                            j  += 2
                            ok = True
                    if i > 0 and ok != True:
                        if self.lee_matrix[i - 2][j] == ct - 1:
                            self.lee_matrix[i - 1][j] = -2
                            ct -= 1
                            i  -= 2
                            ok = True
                    if i < 16 and ok != True:
                        if self.lee_matrix[i + 2][j] == ct - 1:
                            self.lee_matrix[i + 1][j] = -2
                            ct -= 1
                            i  += 2
                            ok = True
                    self.lee_matrix[i][j] = -2
                    
                self.lee_matrix[self.positionLeePlayer[0] * 2 + (i // 2 - self.positionLeePlayer[0])][self.positionLeePlayer[1] * 2 + (j // 2 - self.positionLeePlayer[1])] = 0
                self.positionLeePlayer[0] = i // 2
                self.positionLeePlayer[1] = j // 2
                self.lee_matrix[i][j] = 0
                
                for el in self.lee_matrix:
                    for el1 in el:
                        print(el1, end=' ')
                    print()
        else:
            # we do not change our path, we continue on the last one created
            self.distance_lee -= 1
            i = self.positionLeePlayer[0] * 2
            j = self.positionLeePlayer[1] * 2
            ok = False
            if j > 0 and ok != True:
                if self.lee_matrix[i][j - 2] == -2:
                    j  -= 2
                    ok = True
            if j < 16 and ok != True:
                if self.lee_matrix[i][j + 2] == -2:
                    j  += 2
                    ok = True
            if i > 0 and ok != True:
                if self.lee_matrix[i - 2][j] == -2:
                    i  -= 2
                    ok = True
            if i < 16 and ok != True:
                if self.lee_matrix[i + 2][j] == -2:
                    i  += 2
                    ok = True
                    
            self.lee_matrix[self.positionLeePlayer[0] * 2 + (i // 2 - self.positionLeePlayer[0])][self.positionLeePlayer[1] * 2 + (j // 2 - self.positionLeePlayer[1])] = 0
            self.positionLeePlayer[0] = i // 2
            self.positionLeePlayer[1] = j // 2
            self.lee_matrix[i][j] = 0
            for el in self.lee_matrix:
                for el1 in el:
                    print(el1, end=' ')
                print()


    def check_pawn_action(self, action):
        # move up
        if action == 0:
            posX = self.positionAIPlayer[0] - 1
            if self.positionAIPlayer[0] == 0:
                return False
            if self.positionAIPlayer[1] == 0:
                if self.wallPlaces[posX][0] == 1:
                    return False
            elif self.positionAIPlayer[1] == 8:
                if self.wallPlaces[posX][7] == 1:
                    return False
            elif self.wallPlaces[posX][self.positionAIPlayer[1]] == 1 or self.wallPlaces[posX][self.positionAIPlayer[1] - 1] == 1:
                return False
            
            self.positionAIPlayer[0] -= 1
            return True
        # move down
        if action == 1:
            posX = self.positionAIPlayer[0]
            if self.positionAIPlayer[0] == 8:
                return False
            if self.positionAIPlayer[1] == 0:
                if self.wallPlaces[posX][0] == 1:
                    return False
            elif self.positionAIPlayer[1] == 8:
                if self.wallPlaces[posX][7] == 1:
                    return False
            elif self.wallPlaces[posX][self.positionAIPlayer[1]] == 1 or self.wallPlaces[posX][self.positionAIPlayer[1] - 1] == 1:
                return False
            
            self.positionAIPlayer[0] -= 1
            return True
        # move left
        if action == 2:
            posY = self.positionAIPlayer[1] - 1 
            if self.positionAIPlayer[1] == 0:
                return False
            if self.positionAIPlayer[0] == 0:
                if self.wallPlaces[0][posY] == 2:
                    return False
            elif self.positionAIPlayer[0] == 8: 
                if self.wallPlaces[7][posY] == 2:
                    return False
            elif self.wallPlaces[self.positionAIPlayer[0]][posY] == 2 or self.wallPlaces[self.positionAIPlayer[0] - 1][posY] == 2:
                return False
            self.positionAIPlayer[1] -= 1
            return True
        # move right
        if action == 3:
            posY = self.positionAIPlayer[1]
            if self.positionAIPlayer[1] == 8:
                return False
            if self.positionAIPlayer[0] == 0:
                if self.wallPlaces[0][posY] == 2:
                    return False
            elif self.positionAIPlayer[0] == 8:
                if self.wallPlaces[7][posY] == 2:
                    return False
            elif self.wallPlaces[self.positionAIPlayer[0]][posY] == 2 or self.wallPlaces[self.positionAIPlayer[0] - 1][posY] == 2:
                return False
            self.positionAIPlayer[1] += 1
            return True

    def check_path_existence(self, wall_place_X, wall_place_Y, wall_place_position):
        copy_wallPlaces = []

        for row in self.wallPlaces:
            copy_row = []
            for el in row:
                copy_row.append(el)
            copy_wallPlaces.append(copy_row)

        copy_wallPlaces[wall_place_X][wall_place_Y] = wall_place_position + 1
        visited = []
        ok1 = self.find_path(copy_wallPlaces, visited, self.positionAIPlayer[0], self.positionAIPlayer[1], 0)

        visited = []
        ok2 = self.find_path(copy_wallPlaces, visited, self.positionLeePlayer[0], self.positionLeePlayer[1], 8)

        return ok1 and ok2

    def check_wall_action(self, action):
        wall_place_X = (action - 4) // 2 // 8
        wall_place_Y = (action - 4) // 2  % 8
        wall_place_position = action % 2

        print("Walls allowed: " + str(self.wallsAllowed))
        if self.wallsAllowed <= 0:
            self.reward -= 80
            return False
        if self.wallPlaces[wall_place_X][wall_place_Y] != 0:
            return False


        if wall_place_position == 1:
            if wall_place_X > 0:
                if self.wallPlaces[wall_place_X - 1][wall_place_Y] == 2:
                    return False
            if wall_place_X < 7:
                if self.wallPlaces[wall_place_X + 1][wall_place_Y] == 2:
                    return False
        else:
            if wall_place_Y > 0:
                if self.wallPlaces[wall_place_X][wall_place_Y - 1] == 1:
                    return False
            if wall_place_Y < 7:
                if self.wallPlaces[wall_place_X][wall_place_Y + 1] == 1:
                    return False
        
        if self.check_path_existence(wall_place_X, wall_place_Y, wall_place_position) == False:
            return False
        
        self.wallsAllowed -= 1
        self.wallPlaces[wall_place_X][wall_place_Y] = wall_place_position + 1

        return True

    def step(self, action):
        done = 0
        AI_move = False
        self.reward = 0
        if action <= 3:
            AI_move = self.check_pawn_action(action)
            done = self.check_end()
            if AI_move != True:
                print("Mistake")
                self.reward -= 30
                done = 3
            else:
                self.reward += 30
        else:
            AI_move = self.check_wall_action(action)
            if AI_move != True:
                print("Mistake")
                self.reward -= 40
                done = 3
            else:
                self.reward += 5

        if done == 0 and AI_move:
            self.enemys_move()
            done = self.check_end()

        if done == 1:
            self.reward -= 100
            done = True
        elif done == 2:
            self.reward += 200
            done = True
        else:
            done = False
        
        self.print_matrix()

        wallPlacesOneArray = [self.wallPlaces[i % 8][i // 8] for i in range(64)]
        observation = [self.positionAIPlayer[0], self.positionAIPlayer[1], self.wallsAllowed, self.positionLeePlayer[0], self.positionLeePlayer[1]] + wallPlacesOneArray
        observation = np.array(observation)
        info = {}
        return observation, self.reward, done, info
  
    def reset(self):
        
        self.positionAIPlayer = [8, 4]
        self.positionLeePlayer = [0, 4]
        self.wallsAllowed = 10
        self.lee_matrix = [[0 for _ in range(17)] for _ in range(17)]
        self.reward = 0
        self.distance_lee = -1
        # 0 - no wall
        # 1 - horizontal wall
        # 2 - vertical wall
        self.wallPlaces = [[0 for _ in range(8)] for _ in range(8)]
        
        wallPlacesOneArray = [self.wallPlaces[i % 8][i // 8] for i in range(64)]
        self.observation = [self.positionAIPlayer[0], self.positionAIPlayer[1], self.wallsAllowed, self.positionLeePlayer[0], self.positionLeePlayer[1]] + wallPlacesOneArray
        self.observation = np.array(self.observation)

        return self.observation  # reward, done, info can't be included

    def close (self):
        ...