import gym
import numpy as np
from gym import spaces
import pygame
import heapq
import time
import random


class RobotPlanningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(RobotPlanningEnv, self).__init__()

        self.ekran_genisligi, self.ekran_yuksekligi = 700, 700

        self.observation_space = spaces.Box(low=0, high=500, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # Yukarı, aşağı, sağa, sola

        self.robot_pozisyonu = random.randint(0, 650)
        self.blok_pozisyonu = {"A": [90, 110], "B": [100, 80]}
        self.hedef_pozisyonu = {"A": [300, 300], "B": [400, 300]}
        self.engel_pozisyonlari = [[random.randint(0, 650), random.randint(0, 650)] for _ in range(50)]

        self.tasiniyor_mu = None

        pygame.init()
        self.ekran = pygame.display.set_mode((self.ekran_genisligi, self.ekran_yuksekligi))
        pygame.display.set_caption("Robot Planlama Simülasyonu")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.robot_pozisyonu = [50, 50]
        self.blok_pozisyonu = {"A": [100, 100], "B": [200, 100]}
        self.engel_pozisyonlari = [[random.randint(0, 450), random.randint(0, 450)] for _ in range(5)]
        self.tasiniyor_mu = None
        return np.array(self.robot_pozisyonu, dtype=np.int32)

    def step(self, action):
        if action == 0:  # Yukarı
            self.robot_pozisyonu[1] -= 10
        elif action == 1:  # Aşağı
            self.robot_pozisyonu[1] += 10
        elif action == 2:  # Sağa
            self.robot_pozisyonu[0] += 10
        elif action == 3:  # Sola
            self.robot_pozisyonu[0] -= 10

        self.robot_pozisyonu[0] = np.clip(self.robot_pozisyonu[0], 0, self.ekran_genisligi - 10)
        self.robot_pozisyonu[1] = np.clip(self.robot_pozisyonu[1], 0, self.ekran_yuksekligi - 10)

        for engel in self.engel_pozisyonlari:
            if (self.robot_pozisyonu[0] in range(engel[0], engel[0] + 40) and
                    self.robot_pozisyonu[1] in range(engel[1], engel[1] + 40)):
                reward = -1  # Engel ile çarpışma
                done = True
                return np.array(self.robot_pozisyonu, dtype=np.int32), reward, done, {}

        if self.tasiniyor_mu is None:
            for blok, pozisyon in self.blok_pozisyonu.items():
                if (self.robot_pozisyonu[0] in range(pozisyon[0], pozisyon[0] + 40) and
                        self.robot_pozisyonu[1] in range(pozisyon[1], pozisyon[1] + 40)):
                    self.tasiniyor_mu = blok
                    break
        elif self.tasiniyor_mu:
            self.blok_pozisyonu[self.tasiniyor_mu] = self.robot_pozisyonu[:]
            if self.blok_pozisyonu[self.tasiniyor_mu] == self.hedef_pozisyonu[self.tasiniyor_mu]:
                self.tasiniyor_mu = None

        done = all(self.blok_pozisyonu[blok] == self.hedef_pozisyonu[blok] for blok in self.blok_pozisyonu)
        reward = 1 if done else 0

        return np.array(self.robot_pozisyonu, dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        self.ekran.fill((255, 255, 255))
        pygame.draw.circle(self.ekran, (0, 0, 0), self.robot_pozisyonu, 20)

        for blok, pozisyon in self.blok_pozisyonu.items():
            pygame.draw.rect(self.ekran, (255, 0, 0), (*pozisyon, 40, 40))

        for engel in self.engel_pozisyonlari:
            pygame.draw.rect(self.ekran, (0, 0, 255), (*engel, 40, 40))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def plan_path(self, start, goal):

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while tuple(current) in came_from:
                    path.append(current)
                    current = came_from[tuple(current)]
                path.reverse()
                return path

            neighbors = [
                [current[0] + dx, current[1] + dy]
                for dx, dy in [(-10, 0), (10, 0), (0, -10), (0, 10)]
            ]

            for neighbor in neighbors:
                if not (0 <= neighbor[0] < self.ekran_genisligi and 0 <= neighbor[1] < self.ekran_yuksekligi):
                    continue

                if any(
                        neighbor[0] in range(engel[0], engel[0] + 40) and neighbor[1] in range(engel[1], engel[1] + 40)
                        for engel in self.engel_pozisyonlari
                ):
                    continue

                tentative_g_score = g_score[tuple(current)] + 1

                if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g_score
                    f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))

        return []

env = RobotPlanningEnv()
obs = env.reset()

for blok, hedef in env.hedef_pozisyonu.items():
    path = env.plan_path(env.robot_pozisyonu, env.blok_pozisyonu[blok])
    for step in path:
        env.robot_pozisyonu = step
        env.render()
        time.sleep(0.1)

    env.tasiniyor_mu = blok

    path = env.plan_path(env.robot_pozisyonu, hedef)
    for step in path:
        env.robot_pozisyonu = step
        env.blok_pozisyonu[blok] = step
        env.render()
        time.sleep(0.1)

env.close()
