from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import random
import numpy as np
from gym import spaces
import pygame
import heapq
import time
import gym

class RobotPlanningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, ekran_genisligi=700, ekran_yuksekligi=700, engel_sayisi=50, baslangic_konumu=(50, 50), blok_pozisyonlari=None):
        super(RobotPlanningEnv, self).__init__()

        self.ekran_genisligi = ekran_genisligi
        self.ekran_yuksekligi = ekran_yuksekligi
        self.engel_sayisi = engel_sayisi
        self.baslangic_konumu = list(baslangic_konumu)
        self.blok_pozisyonlari = blok_pozisyonlari or {"A": [100, 100], "B": [200, 100]}

        self.observation_space = spaces.Box(low=0, high=500, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # Yukarı, aşağı, sağa, sola

        self.robot_pozisyonu = self.baslangic_konumu[:]
        self.blok_pozisyonu = self.blok_pozisyonlari
        self.hedef_pozisyonu = {"A": [650, 300], "B": [600, 300]}
        self.engel_pozisyonlari = [[random.randint(0, 650), random.randint(0, 650)] for _ in range(self.engel_sayisi)]

        self.tasiniyor_mu = None

        pygame.init()
        self.ekran = pygame.display.set_mode((self.ekran_genisligi, self.ekran_yuksekligi))
        pygame.display.set_caption("Robot Planlama Simülasyonu")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.robot_pozisyonu = self.baslangic_konumu[:]
        self.blok_pozisyonu = self.blok_pozisyonlari
        self.engel_pozisyonlari = [[random.randint(0, 450), random.randint(0, 450)] for _ in range(self.engel_sayisi)]
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

        # Draw grid
        for x in range(0, self.ekran_genisligi, 50):
            pygame.draw.line(self.ekran, (200, 200, 200), (x, 0), (x, self.ekran_yuksekligi))
        for y in range(0, self.ekran_yuksekligi, 50):
            pygame.draw.line(self.ekran, (200, 200, 200), (0, y), (self.ekran_genisligi, y))

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

class RobotSimulationUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Robot Planlama Simülasyonu")
        self.setGeometry(100, 100, 500, 500)

        # Layouts
        main_layout = QtWidgets.QVBoxLayout()

        # Start Position Inputs
        start_position_layout = QtWidgets.QHBoxLayout()
        start_position_label = QtWidgets.QLabel("Robot Başlangıç Konumu (x, y):")
        self.start_x_input = QtWidgets.QSpinBox()
        self.start_x_input.setRange(0, 700)
        self.start_y_input = QtWidgets.QSpinBox()
        self.start_y_input.setRange(0, 700)
        start_position_layout.addWidget(start_position_label)
        start_position_layout.addWidget(self.start_x_input)
        start_position_layout.addWidget(self.start_y_input)

        # Block Position Inputs
        block_position_layout = QtWidgets.QVBoxLayout()
        block_position_label = QtWidgets.QLabel("Blok Başlangıç Konumları:")

        self.block_a_x_input = QtWidgets.QSpinBox()
        self.block_a_x_input.setRange(0, 700)
        self.block_a_y_input = QtWidgets.QSpinBox()
        self.block_a_y_input.setRange(0, 700)
        block_a_layout = QtWidgets.QHBoxLayout()
        block_a_layout.addWidget(QtWidgets.QLabel("A (x, y):"))
        block_a_layout.addWidget(self.block_a_x_input)
        block_a_layout.addWidget(self.block_a_y_input)

        self.block_b_x_input = QtWidgets.QSpinBox()
        self.block_b_x_input.setRange(0, 700)
        self.block_b_y_input = QtWidgets.QSpinBox()
        self.block_b_y_input.setRange(0, 700)
        block_b_layout = QtWidgets.QHBoxLayout()
        block_b_layout.addWidget(QtWidgets.QLabel("B (x, y):"))
        block_b_layout.addWidget(self.block_b_x_input)
        block_b_layout.addWidget(self.block_b_y_input)

        block_position_layout.addWidget(block_position_label)
        block_position_layout.addLayout(block_a_layout)
        block_position_layout.addLayout(block_b_layout)

        # Number of Obstacles
        obstacle_layout = QtWidgets.QHBoxLayout()
        obstacle_label = QtWidgets.QLabel("Engel Sayısı:")
        self.obstacle_input = QtWidgets.QSpinBox()
        self.obstacle_input.setRange(0, 100)
        obstacle_layout.addWidget(obstacle_label)
        obstacle_layout.addWidget(self.obstacle_input)

        # Start and Stop Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Simülasyonu Başlat")
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button = QtWidgets.QPushButton("Simülasyonu Durdur")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Add layouts to main layout
        main_layout.addLayout(start_position_layout)
        main_layout.addLayout(block_position_layout)
        main_layout.addLayout(obstacle_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def start_simulation(self):
        # Get user inputs
        start_x = self.start_x_input.value()
        start_y = self.start_y_input.value()
        num_obstacles = self.obstacle_input.value()

        block_a_x = self.block_a_x_input.value()
        block_a_y = self.block_a_y_input.value()
        block_b_x = self.block_b_x_input.value()
        block_b_y = self.block_b_y_input.value()

        block_positions = {
            "A": [block_a_x, block_a_y],
            "B": [block_b_x, block_b_y]
        }

        print(f"Simülasyon Başlıyor: Başlangıç Konumu: ({start_x}, {start_y}), Engel Sayısı: {num_obstacles}, Bloklar: {block_positions}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Başlat simülasyon
        self.env = RobotPlanningEnv(engel_sayisi=num_obstacles, baslangic_konumu=(start_x, start_y), blok_pozisyonlari=block_positions)
        obs = self.env.reset()

        for blok, hedef in self.env.hedef_pozisyonu.items():
            path = self.env.plan_path(self.env.robot_pozisyonu, self.env.blok_pozisyonu[blok])
            for step in path:
                self.env.robot_pozisyonu = step
                self.env.render()
                time.sleep(0.1)

            self.env.tasiniyor_mu = blok

            path = self.env.plan_path(self.env.robot_pozisyonu, hedef)
            for step in path:
                self.env.robot_pozisyonu = step
                self.env.blok_pozisyonu[blok] = step
                self.env.render()
                time.sleep(0.1)

    def stop_simulation(self):
        print("Simülasyon Durduruldu.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if hasattr(self, 'env'):
            self.env.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RobotSimulationUI()
    window.show()
    sys.exit(app.exec_())
