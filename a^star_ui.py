import sys
import random
import numpy as np
import pygame
import heapq
import time
import gym
from gym import spaces
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time

# --- Boyut sabitleri ---
BLOCK_SIZE = 50   # Bloklar ve engeller 50×50
ROBOT_SIZE = 30   # Robot kare boyutu: 30×30
ROBOT_OFFSET = 10 # Robotun hücre içindeki ofseti (görsel olarak çizgilerle çakışmaması için)

def boxes_overlap(box1, box2):
    """
    İki eksen-paralel kutunun çakışıp çakışmadığını kontrol eder.
    box = {'left':..., 'right':..., 'top':..., 'bottom':...}
    """
    overlap = not (
        box1['right'] <= box2['left'] or
        box1['left'] >= box2['right'] or
        box1['bottom'] <= box2['top'] or
        box1['top'] >= box2['bottom']
    )
    # Debug: hangi kutuların çakıştığını gösterelim
    if overlap:
        print(f"[DEBUG] Overlap tespit edildi!\n  box1: {box1}\n  box2: {box2}\n")
    return overlap

class RobotPlanningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ekran_genisligi=700,
        ekran_yuksekligi=700,
        engel_sayisi=10,
        baslangic_konumu=(0, 0),
        blok_pozisyonlari=None,
        hedef_pozisyonlari=None
    ):
        super(RobotPlanningEnv, self).__init__()

        self.ekran_genisligi = ekran_genisligi
        self.ekran_yuksekligi = ekran_yuksekligi
        self.engel_sayisi = engel_sayisi

        # Varsayılan blok/hedef konumları
        default_blok_pos = {"A": [100, 100], "B": [200, 100]}
        default_hedef_pos = {"A": [350, 350], "B": [400, 350]}

        # Parametrelerden gelmeyenler varsayılan olsun
        input_blok_pos = blok_pozisyonlari or default_blok_pos
        input_hedef_pos = hedef_pozisyonlari or default_hedef_pos

        # Robot başlangıç konumunu grid'e hizala
        self.baslangic_konumu = self.validate_and_align_position(baslangic_konumu)

        # Blok konumlarını hizala
        self.blok_pozisyonlari = {}
        for blk, pos in input_blok_pos.items():
            aligned = self.validate_and_align_position(pos)
            self.blok_pozisyonlari[blk] = aligned

        # Hedef konumlarını hizala
        self.hedef_pozisyonu = {}
        for blk, pos in input_hedef_pos.items():
            aligned = self.validate_and_align_position(pos)
            self.hedef_pozisyonu[blk] = aligned

        # Gözlem ve aksiyon uzayları
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.ekran_genisligi, self.ekran_yuksekligi]),
            shape=(2,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)  # Yukarı, Aşağı, Sağa, Sola

        # Pygame ayarları
        pygame.init()
        self.setup_display()

        # Ortamı başlat
        print(f"[DEBUG] RobotPlanningEnv başlatıldı.\n"
              f"  Başlangıç konumu: {self.baslangic_konumu}\n"
              f"  Blok konumları: {self.blok_pozisyonlari}\n"
              f"  Hedef konumları: {self.hedef_pozisyonu}\n"
              f"  Engel sayısı: {self.engel_sayisi}\n")
        self.reset()

    def setup_display(self):
        """Pygame ekranını oluştur."""
        self.ekran = pygame.display.set_mode((self.ekran_genisligi, self.ekran_yuksekligi))
        pygame.display.set_caption("Robot Planlama Simülasyonu (30x30 Robot, 50x50 Blok)")
        self.clock = pygame.time.Clock()

    def validate_and_align_position(self, pos):
        """
        Pozisyonu 50 px gridine hizala (sol üst köşe mantığı).
        Örn: (73, 89) => (50, 100)
        """
        x = round(pos[0] / BLOCK_SIZE) * BLOCK_SIZE
        y = round(pos[1] / BLOCK_SIZE) * BLOCK_SIZE
        x = np.clip(x, 0, self.ekran_genisligi - BLOCK_SIZE)
        y = np.clip(y, 0, self.ekran_yuksekligi - BLOCK_SIZE)
        return [x, y]

    def get_box_for_robot(self, pos):
        """
        Robotun (30×30) bounding-box'ını verir. (Ofsetli)
        """
        left = pos[0] + ROBOT_OFFSET
        right = left + ROBOT_SIZE
        top = pos[1] + ROBOT_OFFSET
        bottom = top + ROBOT_SIZE
        return {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom
        }

    def get_box_for_block(self, pos):
        """
        Blok/Engel (50×50) bounding box'ı.
        """
        return {
            'left': pos[0],
            'right': pos[0] + BLOCK_SIZE,
            'top': pos[1],
            'bottom': pos[1] + BLOCK_SIZE
        }

    def check_collisions(self, position, is_carrying_block=False, ignore_block=None):
        """Çarpışma kontrolü."""
        print(f"[DEBUG] Çarpışma kontrolü:")
        print(f"[DEBUG] Kontrol edilen pozisyon: {position}")
        print(f"[DEBUG] Blok taşınıyor mu: {is_carrying_block}")
        print(f"[DEBUG] Yok sayılan blok: {ignore_block}")

        # Engellerle çarpışma kontrolü
        for engel in self.engel_pozisyonlari:
            if position[0] == engel[0] and position[1] == engel[1]:
                print("[DEBUG] Engelle çarpışma tespit edildi")
                return True

        # Bloklarla çarpışma kontrolü
        for blk, pos in self.blok_pozisyonu.items():
            # Bu blok ignore_block veya almaya çalıştığımız blok ise atla
            if blk == ignore_block or blk == self.tasiniyor_mu:
                continue

            # Blok taşımıyorsak ve bloğun yanındaysak, çarpışma sayma
            if not is_carrying_block and pos[0] == position[0] and pos[1] == position[1]:
                continue

            # Robot pozisyonunda blok varsa çarpışma var
            if pos[0] == position[0] and pos[1] == position[1]:
                if not (position[0] == self.hedef_pozisyonu[blk][0] and
                        position[1] == self.hedef_pozisyonu[blk][1]):
                    print(f"[DEBUG] Blok {blk} ile çarpışma tespit edildi")
                    return True

        return False

    def generate_obstacles(self):
        """Engelleri rastgele üret, robot/blok/hedef konumlarıyla çakışmasın."""
        obstacles = []
        max_attempts = 1000
        attempts = 0

        while len(obstacles) < self.engel_sayisi and attempts < max_attempts:
            x = random.randint(0, (self.ekran_genisligi // BLOCK_SIZE) - 1) * BLOCK_SIZE
            y = random.randint(0, (self.ekran_yuksekligi // BLOCK_SIZE) - 1) * BLOCK_SIZE
            pos = [x, y]

            engel_box = self.get_box_for_block(pos)
            robot_box = self.get_box_for_robot(self.baslangic_konumu)

            # Robotla çakışma
            if boxes_overlap(engel_box, robot_box):
                attempts += 1
                continue

            valid = True
            # Bloklar
            for blkpos in self.blok_pozisyonlari.values():
                if boxes_overlap(engel_box, self.get_box_for_block(blkpos)):
                    valid = False
                    break
            # Hedefler
            for hpos in self.hedef_pozisyonu.values():
                if boxes_overlap(engel_box, self.get_box_for_block(hpos)):
                    valid = False
                    break

            if valid:
                obstacles.append(pos)
            attempts += 1

        print(f"[DEBUG] Rastgele üretilen {len(obstacles)} engel konumu:\n  {obstacles}\n")
        return obstacles

    def reset(self):
        """
        Ortamı sıfırla: robotu başlangıca koy, blokları yerine yerleştir, engelleri üret.
        """
        self.robot_pozisyonu = self.baslangic_konumu[:]
        self.blok_pozisyonu = dict(self.blok_pozisyonlari)
        self.engel_pozisyonlari = self.generate_obstacles()
        self.tasiniyor_mu = None
        self.tasinan_bloklar = set()

        print("[DEBUG] Ortam resetlendi.\n"
              f"  Robot pozisyonu: {self.robot_pozisyonu}\n"
              f"  Bloklar: {self.blok_pozisyonu}\n"
              f"  Engeller: {self.engel_pozisyonlari}\n")

        return np.array(self.robot_pozisyonu, dtype=np.int32)

    def step(self, action):
        """Bir adım hareket et."""
        print(f"\n[DEBUG] Step başlangıcı:")
        print(f"[DEBUG] Robot pozisyonu: {self.robot_pozisyonu}")
        print(f"[DEBUG] Taşınan blok: {self.tasiniyor_mu}")
        print(f"[DEBUG] Blok pozisyonları: {self.blok_pozisyonu}")

        # Yeni pozisyonu hesapla
        yeni_pozisyon = self.robot_pozisyonu[:]
        if action == 0:  # Yukarı
            yeni_pozisyon[1] -= BLOCK_SIZE
        elif action == 1:  # Aşağı
            yeni_pozisyon[1] += BLOCK_SIZE
        elif action == 2:  # Sağa
            yeni_pozisyon[0] += BLOCK_SIZE
        elif action == 3:  # Sola
            yeni_pozisyon[0] -= BLOCK_SIZE

        # Grid'e hizala
        yeni_pozisyon = self.validate_and_align_position(yeni_pozisyon)

        # Blok alma/bırakma mantığı
        if self.tasiniyor_mu is None:  # Robot boşta, blok alabilir
            print("[DEBUG] Robot boşta, blok kontrolü yapılıyor")
            for blk, pos in self.blok_pozisyonu.items():
                if (blk not in self.tasinan_bloklar and
                        pos[0] == self.robot_pozisyonu[0] and
                        pos[1] == self.robot_pozisyonu[1]):
                    print(f"[DEBUG] Blok {blk} alındı!")
                    self.tasiniyor_mu = blk
                    break

        # Çarpışma kontrolü
        if self.check_collisions(yeni_pozisyon, is_carrying_block=(self.tasiniyor_mu is not None)):
            print("[DEBUG] Çarpışma var, hareket iptal")
            return np.array(self.robot_pozisyonu), -1, False, {}

        # Hareketi uygula
        self.robot_pozisyonu = yeni_pozisyon

        # Taşınan bloğun pozisyonunu güncelle
        if self.tasiniyor_mu:
            print(f"[DEBUG] Blok {self.tasiniyor_mu} taşınıyor")
            self.blok_pozisyonu[self.tasiniyor_mu] = self.robot_pozisyonu[:]

            # Hedef kontrolü
            hedef = self.hedef_pozisyonu[self.tasiniyor_mu]
            if (self.robot_pozisyonu[0] == hedef[0] and
                    self.robot_pozisyonu[1] == hedef[1]):
                print(f"[DEBUG] Blok {self.tasiniyor_mu} hedefe ulaştı!")
                self.tasinan_bloklar.add(self.tasiniyor_mu)
                self.tasiniyor_mu = None

        done = len(self.tasinan_bloklar) == len(self.blok_pozisyonu)
        reward = 1 if done else 0

        print("[DEBUG] Step sonu:")
        print(f"[DEBUG] Robot pozisyonu: {self.robot_pozisyonu}")
        print(f"[DEBUG] Taşınan blok: {self.tasiniyor_mu}")
        print(f"[DEBUG] Blok pozisyonları: {self.blok_pozisyonu}")

        return np.array(self.robot_pozisyonu), reward, done, {}

    def render(self, mode="human"):
        """Pygame ile ortamı çiz."""
        self.ekran.fill((255, 255, 255))

        # Grid çizgileri
        for x in range(0, self.ekran_genisligi, BLOCK_SIZE):
            pygame.draw.line(self.ekran, (200, 200, 200), (x, 0), (x, self.ekran_yuksekligi))
        for y in range(0, self.ekran_yuksekligi, BLOCK_SIZE):
            pygame.draw.line(self.ekran, (200, 200, 200), (0, y), (self.ekran_genisligi, y))

        # Engeller (mavi)
        for engel in self.engel_pozisyonlari:
            pygame.draw.rect(self.ekran, (0, 0, 255),
                             (engel[0], engel[1], BLOCK_SIZE, BLOCK_SIZE))

        # Hedefler (yeşil)
        for hf in self.hedef_pozisyonu.values():
            pygame.draw.rect(self.ekran, (0, 255, 0),
                             (hf[0], hf[1], BLOCK_SIZE, BLOCK_SIZE))

        # Bloklar (kırmızı)
        for blk, pos in self.blok_pozisyonu.items():
            pygame.draw.rect(self.ekran, (255, 0, 0),
                             (pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))

        # Robot (siyah)
        rx = self.robot_pozisyonu[0] + ROBOT_OFFSET
        ry = self.robot_pozisyonu[1] + ROBOT_OFFSET
        pygame.draw.rect(self.ekran, (0, 0, 0), (rx, ry, ROBOT_SIZE, ROBOT_SIZE))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        """Pygame kapat."""
        if hasattr(self, 'ekran'):
            pygame.quit()

    def plan_path(self, start, goal, target_block=None):
        """
        A* ile (start->goal) arası yol bulur.
        target_block: Gideceğimiz bloksa, onu engel sayma.
        """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): heuristic(start, goal)}

        print(f"[DEBUG] A* plan_path() başlıyor:\n"
              f"  start={start}, goal={goal}, ignore_block={target_block}\n")

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # Yol bulundu
                path = []
                while tuple(current) in came_from:
                    path.append(current)
                    current = came_from[tuple(current)]
                path.append(start)
                path.reverse()
                print(f"[DEBUG] Yol bulundu: {path}\n")
                return path

            directions = [(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0),
                          (0, -BLOCK_SIZE), (0, BLOCK_SIZE)]
            for dx, dy in directions:
                neighbor = [current[0] + dx, current[1] + dy]

                # Sınır kontrolü
                if not (0 <= neighbor[0] <= self.ekran_genisligi - BLOCK_SIZE and
                        0 <= neighbor[1] <= self.ekran_yuksekligi - BLOCK_SIZE):
                    continue

                # Çarpışma kontrolü
                if self.check_collisions(neighbor, is_carrying_block=False, ignore_block=target_block):
                    continue

                tentative_g = g_score[tuple(current)] + 1
                if (tuple(neighbor) not in g_score) or (tentative_g < g_score[tuple(neighbor)]):
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g
                    f_score[tuple(neighbor)] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))

        print(f"[DEBUG] Yol bulunamadı! Start: {start}, Goal: {goal}\n")
        return []

class RobotSimulationUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        print("[DEBUG] RobotSimulationUI init ediliyor...")
        self.init_ui()
        self.simulation_active = False
        self.env = None

    def init_ui(self):
        self.setWindowTitle("Robot Planlama Simülasyonu - PyQt Arayüzü")
        self.setGeometry(100, 100, 500, 500)

        main_layout = QtWidgets.QVBoxLayout()

        # --- 1) Robot Başlangıç Konumu ---
        start_position_layout = QtWidgets.QHBoxLayout()
        start_position_label = QtWidgets.QLabel("Robot Başlangıç Konumu (x, y):")

        self.start_x_input = QtWidgets.QSpinBox()
        self.start_x_input.setRange(0, 650)
        self.start_x_input.setValue(0)      # Varsayılan 0
        self.start_x_input.setSingleStep(50)

        self.start_y_input = QtWidgets.QSpinBox()
        self.start_y_input.setRange(0, 650)
        self.start_y_input.setValue(0)      # Varsayılan 0
        self.start_y_input.setSingleStep(50)

        start_position_layout.addWidget(start_position_label)
        start_position_layout.addWidget(self.start_x_input)
        start_position_layout.addWidget(self.start_y_input)

        # --- 2) Blok Başlangıç Konumları ---
        block_position_layout = QtWidgets.QVBoxLayout()
        block_position_label = QtWidgets.QLabel("Blok Başlangıç Konumları:")

        # Blok A
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

        # Blok B
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

        # --- 3) Hedef Konumları ---
        goal_position_layout = QtWidgets.QVBoxLayout()
        goal_position_label = QtWidgets.QLabel("Hedef Konumları:")

        # Hedef A
        self.goal_a_x_input = QtWidgets.QSpinBox()
        self.goal_a_x_input.setRange(0, 650)
        self.goal_a_x_input.setValue(350)
        self.goal_a_x_input.setSingleStep(50)

        self.goal_a_y_input = QtWidgets.QSpinBox()
        self.goal_a_y_input.setRange(0, 650)
        self.goal_a_y_input.setValue(350)
        self.goal_a_y_input.setSingleStep(50)

        goal_a_layout = QtWidgets.QHBoxLayout()
        goal_a_layout.addWidget(QtWidgets.QLabel("A (x, y):"))
        goal_a_layout.addWidget(self.goal_a_x_input)
        goal_a_layout.addWidget(self.goal_a_y_input)

        # Hedef B
        self.goal_b_x_input = QtWidgets.QSpinBox()
        self.goal_b_x_input.setRange(0, 650)
        self.goal_b_x_input.setValue(400)
        self.goal_b_x_input.setSingleStep(50)

        self.goal_b_y_input = QtWidgets.QSpinBox()
        self.goal_b_y_input.setRange(0, 650)
        self.goal_b_y_input.setValue(350)
        self.goal_b_y_input.setSingleStep(50)

        goal_b_layout = QtWidgets.QHBoxLayout()
        goal_b_layout.addWidget(QtWidgets.QLabel("B (x, y):"))
        goal_b_layout.addWidget(self.goal_b_x_input)
        goal_b_layout.addWidget(self.goal_b_y_input)

        goal_position_layout.addWidget(goal_position_label)
        goal_position_layout.addLayout(goal_a_layout)
        goal_position_layout.addLayout(goal_b_layout)

        # --- 4) Engel Sayısı ---
        obstacle_layout = QtWidgets.QHBoxLayout()
        obstacle_label = QtWidgets.QLabel("Engel Sayısı:")
        self.obstacle_input = QtWidgets.QSpinBox()
        self.obstacle_input.setRange(0, 100)
        self.obstacle_input.setValue(10)  # Varsayılan 10
        obstacle_layout.addWidget(obstacle_label)
        obstacle_layout.addWidget(self.obstacle_input)

        # --- 5) Butonlar ---
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Simülasyonu Başlat")
        self.start_button.clicked.connect(self.start_simulation)

        self.stop_button = QtWidgets.QPushButton("Simülasyonu Durdur")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # --- Layout'ları birleştir ---
        main_layout.addLayout(start_position_layout)
        main_layout.addLayout(block_position_layout)
        main_layout.addLayout(goal_position_layout)
        main_layout.addLayout(obstacle_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def start_simulation(self):
        """Simülasyonu başlatır."""
        if not self.simulation_active:
            print("[DEBUG] Simülasyon başlatılıyor...")
            try:
                self.simulation_active = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)

                # Kullanıcı girişlerini al
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

                print(f"[DEBUG] Kullanıcı girdileri:\n"
                      f"  Başlangıç: ({start_x}, {start_y})\n"
                      f"  Bloklar: {block_positions}\n"
                      f"  Hedefler: {goal_positions}\n"
                      f"  Engel sayısı: {num_obstacles}\n")

                # Eğer env önceden açıksa, kapat
                if self.env:
                    self.env.close()
                    self.env = None

                # Yeni ortam oluştur
                self.env = RobotPlanningEnv(
                    ekran_genisligi=700,
                    ekran_yuksekligi=700,
                    engel_sayisi=num_obstacles,
                    baslangic_konumu=(start_x, start_y),
                    blok_pozisyonlari=block_positions,
                    hedef_pozisyonlari=goal_positions
                )
                self.env.reset()
                self.env.render()

                # Blokları teker teker hedefe taşı
                for blok in self.env.hedef_pozisyonu.keys():
                    if not self.simulation_active:
                        break

                    print(f"[DEBUG] '{blok}' bloğu için hareket başlıyor...")
                    # 1) Bloğa git
                    start_pos = list(self.env.robot_pozisyonu)
                    block_pos = list(self.env.blok_pozisyonu[blok])

                    path_to_block = self.env.plan_path(
                        start=start_pos,
                        goal=block_pos,
                        target_block=blok
                    )

                    if not path_to_block:
                        print(f"[DEBUG] Blok '{blok}' için yol bulunamadı, devam ediliyor.")
                        continue

                    print(f"[DEBUG] Blok '{blok}' yol planı:\n  {path_to_block}\n")

                    # Adım adım ilerle
                    for i in range(len(path_to_block) - 1):
                        if not self.simulation_active:
                            break

                        current_pos = path_to_block[i]
                        next_pos = path_to_block[i + 1]
                        action = self.get_action(current_pos, next_pos)

                        print(f"[DEBUG] Blok '{blok}' -> current:{current_pos}, next:{next_pos}, action:{action}")

                        if action is None:
                            print(f"[DEBUG] Geçersiz hareket: {current_pos} -> {next_pos}")
                            break

                        obs, reward, done, _ = self.env.step(action)
                        self.env.render()
                        time.sleep(0.15)

                        if done:
                            print("[DEBUG] Görev tamamlandı (tüm bloklar yerleştirildi)!")
                            break

                    if not self.simulation_active or done:
                        break

                    # 2) Bloğu hedefe götür
                    goal_pos = self.env.hedef_pozisyonu[blok]
                    path_to_goal = self.env.plan_path(
                        start=self.env.robot_pozisyonu,
                        goal=goal_pos,
                        target_block=blok  # Taşıdığımız blok, planlamada engel sayılmasın
                    )

                    if not path_to_goal:
                        print(f"[DEBUG] Blok '{blok}' için hedefe yol bulunamadı, devam ediliyor.")
                        continue

                    print(f"[DEBUG] Blok '{blok}' hedef yol planı:\n  {path_to_goal}\n")

                    for i in range(len(path_to_goal) - 1):
                        if not self.simulation_active:
                            break

                        current_pos = path_to_goal[i]
                        next_pos = path_to_goal[i + 1]
                        action = self.get_action(current_pos, next_pos)

                        print(f"[DEBUG] Blok '{blok}' hedefe -> current:{current_pos}, next:{next_pos}, action:{action}")

                        if action is None:
                            print(f"[DEBUG] Geçersiz hareket: {current_pos} -> {next_pos}")
                            break

                        obs, reward, done, _ = self.env.step(action)
                        self.env.render()
                        time.sleep(0.15)

                        if done:
                            print("[DEBUG] Görev tamamlandı (tüm bloklar yerleştirildi)!")
                            break

                    if done:
                        break

            except Exception as e:
                print(f"[DEBUG] Hata oluştu: {e}")
            finally:
                # Eğer hâlâ aktif görünüyorsa
                if self.simulation_active:
                    self.stop_simulation()

    def stop_simulation(self):
        """Simülasyonu durdurma."""
        print("[DEBUG] Simülasyon durduruluyor...")
        self.simulation_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.env:
            self.env.close()
            self.env = None

    @staticmethod
    def get_action(current_pos, next_pos):
        """
        Robotun sol üst koordinatında 50 px gridde bir adımlık farkı (yukarı, aşağı, sağ, sol) anla.
        current_pos = (x1, y1)
        next_pos = (x2, y2)

        Dönüş: 0=Yukarı, 1=Aşağı, 2=Sağa, 3=Sola veya None
        """
        cx, cy = current_pos
        nx, ny = next_pos

        # Yukarı
        if nx == cx and ny == cy - 50:
            return 0
        # Aşağı
        elif nx == cx and ny == cy + 50:
            return 1
        # Sağa
        elif nx == cx + 50 and ny == cy:
            return 2
        # Sola
        elif nx == cx - 50 and ny == cy:
            return 3

        # Eğer bu nokta geldiyse, hareket 50px'lik gridde değil demektir
        return None

# Eğer bu script doğrudan çalıştırılırsa (örn. python main.py), arayüzü başlat:
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RobotSimulationUI()
    window.show()
    sys.exit(app.exec_())