from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import random
import numpy as np
from gym import spaces
import pygame
import heapq
import time
import gym


def boxes_overlap(box1, box2):
    return not (
            box1['right'] < box2['left'] or
            box1['left'] > box2['right'] or
            box1['bottom'] < box2['top'] or
            box1['top'] > box2['bottom']
    )


class RobotPlanningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            ekran_genisligi=700,
            ekran_yuksekligi=700,
            engel_sayisi=50,
            baslangic_konumu=(50, 50),
            blok_pozisyonlari=None,
            hedef_pozisyonlari=None
    ):
        super(RobotPlanningEnv, self).__init__()

        self.ekran_genisligi = ekran_genisligi
        self.ekran_yuksekligi = ekran_yuksekligi
        self.engel_sayisi = engel_sayisi

        # Grid hizalama ve sınır kontrolü
        self.baslangic_konumu = self.validate_and_align_position(baslangic_konumu)

        # Blok pozisyonlarını kontrol ve hizalama
        self.blok_pozisyonlari = {}
        default_blok_pos = {"A": [100, 100], "B": [200, 100]}
        input_blok_pos = blok_pozisyonlari or default_blok_pos

        for k, v in input_blok_pos.items():
            pos = self.validate_and_align_position(v)
            if not self.is_position_valid(pos):
                raise ValueError(f"Blok {k} için geçersiz pozisyon: {v}")
            self.blok_pozisyonlari[k] = pos

        # Hedef pozisyonlarını kontrol ve hizalama
        self.hedef_pozisyonu = {}
        default_hedef_pos = {"A": [650, 300], "B": [600, 300]}
        input_hedef_pos = hedef_pozisyonlari or default_hedef_pos

        for k, v in input_hedef_pos.items():
            pos = self.validate_and_align_position(v)
            if not self.is_position_valid(pos):
                raise ValueError(f"Hedef {k} için geçersiz pozisyon: {v}")
            self.hedef_pozisyonu[k] = pos

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.ekran_genisligi, self.ekran_yuksekligi]),
            shape=(2,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        # Pygame başlatma
        pygame.init()
        self.setup_display()
        self.reset()

    def setup_display(self):
        """Pygame ekranını ayarla"""
        self.ekran = pygame.display.set_mode((self.ekran_genisligi, self.ekran_yuksekligi))
        pygame.display.set_caption("Robot Planlama Simülasyonu")
        self.clock = pygame.time.Clock()

    def validate_and_align_position(self, pos):
        """Pozisyonu grid'e hizala ve sınırlar içinde olduğunu kontrol et"""
        x = round(pos[0] / 50) * 50
        y = round(pos[1] / 50) * 50
        x = np.clip(x, 0, self.ekran_genisligi - 50)
        y = np.clip(y, 0, self.ekran_yuksekligi - 50)
        return [x, y]

    def is_position_valid(self, pos):
        """Pozisyonun geçerli olup olmadığını kontrol et"""
        return (0 <= pos[0] <= self.ekran_genisligi - 50 and
                0 <= pos[1] <= self.ekran_yuksekligi - 50)

    def get_box(self, pos, size=50):
        """Verilen pozisyon için çarpışma kutusu oluştur"""
        return {
            'left': pos[0],
            'right': pos[0] + size,
            'top': pos[1],
            'bottom': pos[1] + size
        }

    def check_collisions(self, position, is_carrying_block=False):
        """Çarpışma kontrolü yap"""
        # Robot için çarpışma kutusu
        robot_box = {
            'left': position[0] - 25,
            'right': position[0] + 25,
            'top': position[1] - 25,
            'bottom': position[1] + 25
        }

        # Taşınan blok için çarpışma kutusu
        if is_carrying_block and self.tasiniyor_mu:
            block_box = self.get_box([position[0] - 25, position[1] - 25])
        else:
            block_box = None

        # Engeller ile çarpışma kontrolü
        for engel in self.engel_pozisyonlari:
            engel_box = self.get_box(engel)
            if boxes_overlap(robot_box, engel_box):
                return True
            if block_box and boxes_overlap(block_box, engel_box):
                return True

        # Diğer bloklarla çarpışma kontrolü
        for blok, blok_pos in self.blok_pozisyonu.items():
            if self.tasiniyor_mu != blok:  # Taşınan blok hariç
                blok_box = self.get_box(blok_pos)
                if boxes_overlap(robot_box, blok_box):
                    if not is_carrying_block:  # Blok almaya çalışıyorsak sorun değil
                        return False
                    return True
                if block_box and boxes_overlap(block_box, blok_box):
                    return True

        return False

    def generate_obstacles(self):
        """Engelleri oluştur ve çakışmaları kontrol et"""
        obstacles = []
        max_attempts = 1000
        attempts = 0

        while len(obstacles) < self.engel_sayisi and attempts < max_attempts:
            x = random.randint(0, (self.ekran_genisligi // 50) - 1) * 50
            y = random.randint(0, (self.ekran_yuksekligi // 50) - 1) * 50
            pos = [x, y]

            # Engelin diğer nesnelerle çakışıp çakışmadığını kontrol et
            engel_box = self.get_box(pos)

            # Robot başlangıç pozisyonu kontrolü
            robot_box = {
                'left': self.baslangic_konumu[0] - 25,
                'right': self.baslangic_konumu[0] + 25,
                'top': self.baslangic_konumu[1] - 25,
                'bottom': self.baslangic_konumu[1] + 25
            }

            if boxes_overlap(engel_box, robot_box):
                attempts += 1
                continue

            # Blok ve hedef pozisyonları kontrolü
            valid = True
            for blok_pos in self.blok_pozisyonu.values():
                if boxes_overlap(engel_box, self.get_box(blok_pos)):
                    valid = False
                    break

            for hedef_pos in self.hedef_pozisyonu.values():
                if boxes_overlap(engel_box, self.get_box(hedef_pos)):
                    valid = False
                    break

            if valid:
                obstacles.append(pos)

            attempts += 1

        return obstacles

    def reset(self):
        """Ortamı sıfırla ve başlangıç durumuna getir"""
        self.robot_pozisyonu = self.baslangic_konumu[:]
        self.blok_pozisyonu = dict(self.blok_pozisyonlari)
        self.engel_pozisyonlari = self.generate_obstacles()
        self.tasiniyor_mu = None
        self.tasinan_bloklar = set()
        return np.array(self.robot_pozisyonu, dtype=np.int32)

    def step(self, action):
        """Bir adım ilerle ve sonuçları döndür"""
        eski_pozisyon = self.robot_pozisyonu[:]
        yeni_pozisyon = self.robot_pozisyonu[:]

        # Hareket yönünü belirle
        if action == 0:  # Yukarı
            yeni_pozisyon[1] -= 50
        elif action == 1:  # Aşağı
            yeni_pozisyon[1] += 50
        elif action == 2:  # Sağa
            yeni_pozisyon[0] += 50
        elif action == 3:  # Sola
            yeni_pozisyon[0] -= 50

        # Sınırlar içinde tut
        yeni_pozisyon = self.validate_and_align_position(yeni_pozisyon)

        # Çarpışma kontrolü
        if self.check_collisions(yeni_pozisyon, bool(self.tasiniyor_mu)):
            return np.array(self.robot_pozisyonu, dtype=np.int32), -1, False, {}

        # Hareketi uygula
        self.robot_pozisyonu = yeni_pozisyon

        # Blok taşıma mantığı
        if self.tasiniyor_mu is None:
            # Blok alma kontrolü
            for blok, pozisyon in self.blok_pozisyonu.items():
                if blok not in self.tasinan_bloklar:
                    robot_box = {
                        'left': self.robot_pozisyonu[0] - 25,
                        'right': self.robot_pozisyonu[0] + 25,
                        'top': self.robot_pozisyonu[1] - 25,
                        'bottom': self.robot_pozisyonu[1] + 25
                    }
                    block_box = self.get_box(pozisyon)
                    if boxes_overlap(robot_box, block_box):
                        self.tasiniyor_mu = blok
                        break
        else:
            # Taşınan bloğun pozisyonunu robotun pozisyonuna göre güncelle
            self.blok_pozisyonu[self.tasiniyor_mu] = [
                self.robot_pozisyonu[0] - 25,
                self.robot_pozisyonu[1] - 25
            ]

            # Robot hedef pozisyonuna ulaştıysa bloğu hedefe yerleştir
            if self.robot_pozisyonu == self.hedef_pozisyonu[self.tasiniyor_mu]:
                # Bloğun tam olarak hedef pozisyonuna yerleştirilmesi
                self.blok_pozisyonu[self.tasiniyor_mu] = self.hedef_pozisyonu[self.tasiniyor_mu]
                self.tasinan_bloklar.add(self.tasiniyor_mu)
                self.tasiniyor_mu = None

        # Bütün bloklar hedefte mi kontrolü
        done = all(
            self.blok_pozisyonu[blok] == self.hedef_pozisyonu[blok]
            for blok in self.blok_pozisyonu
        )

        reward = 1 if done else 0

        return np.array(self.robot_pozisyonu, dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        """Ortamı görselleştir"""
        self.ekran.fill((255, 255, 255))

        # Grid çizimi
        for x in range(0, self.ekran_genisligi, 50):
            pygame.draw.line(self.ekran, (200, 200, 200), (x, 0), (x, self.ekran_yuksekligi))
        for y in range(0, self.ekran_yuksekligi, 50):
            pygame.draw.line(self.ekran, (200, 200, 200), (0, y), (self.ekran_genisligi, y))

        # Engeller
        for engel in self.engel_pozisyonlari:
            pygame.draw.rect(self.ekran, (0, 0, 255), (*engel, 50, 50))

        # Hedefler (artık üçgen değil kare)
        for hedef in self.hedef_pozisyonu.values():
            pygame.draw.rect(self.ekran, (0, 255, 0), (*hedef, 50, 50))

        # Bloklar
        for blok, pozisyon in self.blok_pozisyonu.items():
            pygame.draw.rect(self.ekran, (255, 0, 0), (*pozisyon, 50, 50))

        # Robot
        pygame.draw.circle(self.ekran, (0, 0, 0), self.robot_pozisyonu, 25)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        """Kaynakları temizle"""
        if hasattr(self, 'ekran'):
            pygame.quit()

    def plan_path(self, start, goal):
        """A* ile yol planla"""

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
                path.append(start)
                path.reverse()
                return path

            # Olası hareketler
            neighbors = [
                [current[0] + dx, current[1] + dy]
                for dx, dy in [(-50, 0), (50, 0), (0, -50), (0, 50)]
            ]

            for neighbor in neighbors:
                if not (0 <= neighbor[0] < self.ekran_genisligi and
                        0 <= neighbor[1] < self.ekran_yuksekligi):
                    continue

                # Çarpışma kontrolü
                if self.check_collisions(neighbor, bool(self.tasiniyor_mu)):
                    print(f"Çakışma: {neighbor} noktasına gidilemiyor.")
                    continue

                tentative_g_score = g_score[tuple(current)] + 1

                if (tuple(neighbor) not in g_score or
                    tentative_g_score < g_score[tuple(neighbor)]):
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g_score
                    f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))

        print(f"Path bulunamadı: Start: {start}, Goal: {goal}")
        return []  # Yol bulunamadıysa boş liste döndür


class RobotSimulationUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.simulation_active = False

    def init_ui(self):
        self.setWindowTitle("Robot Planlama Simülasyonu")
        self.setGeometry(100, 100, 500, 500)

        main_layout = QtWidgets.QVBoxLayout()

        # Start Position Inputs
        start_position_layout = QtWidgets.QHBoxLayout()
        start_position_label = QtWidgets.QLabel("Robot Başlangıç Konumu (x, y):")
        self.start_x_input = QtWidgets.QSpinBox()
        self.start_x_input.setRange(0, 650)
        self.start_x_input.setValue(650)
        self.start_x_input.setSingleStep(50)
        self.start_y_input = QtWidgets.QSpinBox()
        self.start_y_input.setRange(0, 650)
        self.start_y_input.setValue(0)
        self.start_y_input.setSingleStep(50)
        start_position_layout.addWidget(start_position_label)
        start_position_layout.addWidget(self.start_x_input)
        start_position_layout.addWidget(self.start_y_input)

        # Block Position Inputs
        block_position_layout = QtWidgets.QVBoxLayout()
        block_position_label = QtWidgets.QLabel("Blok Başlangıç Konumları:")

        self.block_a_x_input = QtWidgets.QSpinBox()
        self.block_a_x_input.setRange(0, 650)
        self.block_a_x_input.setValue(100)
        self.block_a_x_input.setSingleStep(50)
        self.block_a_y_input = QtWidgets.QSpinBox()
        self.block_a_y_input.setRange(0, 650)
        self.block_a_y_input.setValue(100)
        self.block_a_y_input.setSingleStep(50)
        block_a_layout = QtWidgets.QHBoxLayout()
        block_a_layout.addWidget(QtWidgets.QLabel("A (x, y):"))
        block_a_layout.addWidget(self.block_a_x_input)
        block_a_layout.addWidget(self.block_a_y_input)

        self.block_b_x_input = QtWidgets.QSpinBox()
        self.block_b_x_input.setRange(0, 650)
        self.block_b_x_input.setValue(200)
        self.block_b_x_input.setSingleStep(50)
        self.block_b_y_input = QtWidgets.QSpinBox()
        self.block_b_y_input.setRange(0, 650)
        self.block_b_y_input.setValue(100)
        self.block_b_y_input.setSingleStep(50)
        block_b_layout = QtWidgets.QHBoxLayout()
        block_b_layout.addWidget(QtWidgets.QLabel("B (x, y):"))
        block_b_layout.addWidget(self.block_b_x_input)
        block_b_layout.addWidget(self.block_b_y_input)

        block_position_layout.addWidget(block_position_label)
        block_position_layout.addLayout(block_a_layout)
        block_position_layout.addLayout(block_b_layout)

        # Goal Position Inputs
        goal_position_layout = QtWidgets.QVBoxLayout()
        goal_position_label = QtWidgets.QLabel("Hedef Konumları:")

        self.goal_a_x_input = QtWidgets.QSpinBox()
        self.goal_a_x_input.setRange(0, 650)
        self.goal_a_x_input.setValue(650)
        self.goal_a_x_input.setSingleStep(50)
        self.goal_a_y_input = QtWidgets.QSpinBox()
        self.goal_a_y_input.setRange(0, 650)
        self.goal_a_y_input.setValue(300)
        self.goal_a_y_input.setSingleStep(50)
        goal_a_layout = QtWidgets.QHBoxLayout()
        goal_a_layout.addWidget(QtWidgets.QLabel("A (x, y):"))
        goal_a_layout.addWidget(self.goal_a_x_input)
        goal_a_layout.addWidget(self.goal_a_y_input)

        self.goal_b_x_input = QtWidgets.QSpinBox()
        self.goal_b_x_input.setRange(0, 650)
        self.goal_b_x_input.setValue(600)
        self.goal_b_x_input.setSingleStep(50)
        self.goal_b_y_input = QtWidgets.QSpinBox()
        self.goal_b_y_input.setRange(0, 650)
        self.goal_b_y_input.setValue(300)
        self.goal_b_y_input.setSingleStep(50)
        goal_b_layout = QtWidgets.QHBoxLayout()
        goal_b_layout.addWidget(QtWidgets.QLabel("B (x, y):"))
        goal_b_layout.addWidget(self.goal_b_x_input)
        goal_b_layout.addWidget(self.goal_b_y_input)

        goal_position_layout.addWidget(goal_position_label)
        goal_position_layout.addLayout(goal_a_layout)
        goal_position_layout.addLayout(goal_b_layout)

        # Obstacle Input
        obstacle_layout = QtWidgets.QHBoxLayout()
        obstacle_label = QtWidgets.QLabel("Engel Sayısı:")
        self.obstacle_input = QtWidgets.QSpinBox()
        self.obstacle_input.setRange(0, 100)
        self.obstacle_input.setValue(30)  # Engel sayısını 10'a düşürdük
        obstacle_layout.addWidget(obstacle_label)
        obstacle_layout.addWidget(self.obstacle_input)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Simülasyonu Başlat")
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button = QtWidgets.QPushButton("Simülasyonu Durdur")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Layout'ları ana layout'a ekle
        main_layout.addLayout(start_position_layout)
        main_layout.addLayout(block_position_layout)
        main_layout.addLayout(goal_position_layout)
        main_layout.addLayout(obstacle_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.env = None

    def start_simulation(self):
        if not self.simulation_active:
            try:
                self.simulation_active = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)

                # Input değerlerini al
                start_x = self.start_x_input.value()
                start_y = self.start_y_input.value()
                num_obstacles = self.obstacle_input.value()

                block_positions = {
                    "A": [self.block_a_x_input.value(), self.block_a_y_input.value()],
                    "B": [self.block_b_x_input.value(), self.block_b_y_input.value()]
                }

                goal_positions = {
                    "A": [self.goal_a_x_input.value(), self.goal_a_y_input.value()],
                    "B": [self.goal_b_x_input.value(), self.goal_b_y_input.value()]
                }

                # Önceki ortamı kapat
                if self.env:
                    self.env.close()

                # Yeni ortam oluştur
                self.env = RobotPlanningEnv(
                    engel_sayisi=num_obstacles,
                    baslangic_konumu=(start_x, start_y),
                    blok_pozisyonlari=block_positions,
                    hedef_pozisyonlari=goal_positions
                )
                self.env.reset()
                self.env.render()

                # Blokları sırayla taşı
                for blok in self.env.hedef_pozisyonu.keys():
                    if not self.simulation_active:
                        break

                    # 1) Bloğa git
                    start_pos = list(self.env.robot_pozisyonu)
                    block_pos = list(self.env.blok_pozisyonu[blok])
                    path_to_block = self.env.plan_path(start_pos, block_pos)

                    if not path_to_block:
                        print(f"Blok {blok}'a ulaşılamıyor!")
                        continue

                    for i in range(len(path_to_block) - 1):
                        if not self.simulation_active:
                            break
                        current_pos = path_to_block[i]
                        next_pos = path_to_block[i + 1]
                        action = self.get_action(current_pos, next_pos)
                        if action is None:
                            print(f"Geçersiz hareket: {current_pos} -> {next_pos}")
                            break
                        obs, reward, done, _ = self.env.step(action)
                        self.env.render()
                        time.sleep(0.15)
                        if done:
                            break

                    if not self.simulation_active or done:
                        break

                    # 2) Hedef noktaya git
                    goal_pos = self.env.hedef_pozisyonu[blok]
                    path_to_goal = self.env.plan_path(self.env.robot_pozisyonu, goal_pos)

                    if not path_to_goal:
                        print(f"Blok {blok} için hedefe ulaşılamıyor!")
                        continue

                    for i in range(len(path_to_goal) - 1):
                        if not self.simulation_active:
                            break
                        current_pos = path_to_goal[i]
                        next_pos = path_to_goal[i + 1]
                        action = self.get_action(current_pos, next_pos)
                        if action is None:
                            print(f"Geçersiz hareket: {current_pos} -> {next_pos}")
                            break
                        obs, reward, done, _ = self.env.step(action)
                        self.env.render()
                        time.sleep(0.15)
                        if done:
                            print("Görev tamamlandı!")
                            break

                    if not self.simulation_active or done:
                        break

            except Exception as e:
                print(f"Hata oluştu: {e}")
                self.stop_simulation()
            finally:
                if self.simulation_active:
                    self.stop_simulation()

    def stop_simulation(self):
        self.simulation_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.env:
            self.env.close()
            self.env = None

    @staticmethod
    def get_action(current_pos, next_pos):
        """50 px'lik adımlarda hareket yönünü belirle"""
        cx, cy = current_pos
        nx, ny = next_pos
        if nx == cx and ny == cy - 50:
            return 0  # yukarı
        elif nx == cx and ny == cy + 50:
            return 1  # aşağı
        elif nx == cx + 50 and ny == cy:
            return 2  # sağa
        elif nx == cx - 50 and ny == cy:
            return 3  # sola
        return None


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RobotSimulationUI()
    window.show()
    sys.exit(app.exec_())
