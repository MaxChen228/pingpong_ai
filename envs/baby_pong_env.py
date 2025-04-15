import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
import random

# <-- 新增匯入 -->
from envs.physics import collide_sphere_with_moving_plane

class BabyPongBounceEnv(gym.Env):
    def __init__(self,
                 move_penalty=0.01,
                 center_penalty_weight=0.03,
                 bounce_reward=1.0,
                 miss_penalty=1.0,
                 perfect_hit_bonus=0.5,
                 max_bounces=100,
                 base_speed=0.02,
                 speed_increment=0.005,
                 speed_scale_every=3,
                 paddle_speed=0.03,
                 paddle_width=60,
                 render_size=400,
                 enable_spin=True,
                 magnus_factor=0.01,
                 # 以下是新增的物理參數
                 restitution=0.9,   # e: 恢復係數 (彈力係數)
                 friction=0.2,      # mu: 摩擦係數
                 ball_mass=1.0,     # 球質量 (kg) - 任意設定
                 world_ball_radius=0.03  # 球在世界座標下的半徑(0~1)規模
                 ):
        super(BabyPongBounceEnv, self).__init__()

        self.move_penalty = move_penalty
        self.center_penalty_weight = center_penalty_weight
        self.bounce_reward = bounce_reward
        self.miss_penalty = miss_penalty
        self.perfect_hit_bonus = perfect_hit_bonus

        self.base_speed = base_speed
        self.speed_increment = speed_increment
        self.speed_scale_every = speed_scale_every
        self.paddle_speed = paddle_speed
        self.max_bounces = max_bounces

        self.window_size = render_size
        self.paddle_width = paddle_width
        self.paddle_height = 10  # 擋板厚度 (畫面像素)
        self.ball_radius = 10    # 球像素半徑 (純視覺用，碰撞計算用 world_ball_radius)

        self.enable_spin = enable_spin
        self.magnus_factor = magnus_factor

        # -- 新增或存取物理參數 --
        self.restitution = restitution
        self.friction = friction
        self.ball_mass = ball_mass
        self.world_ball_radius = world_ball_radius

        self._init_pygame()

        # 觀測空間：ball_x, ball_y, ball_vx, ball_vy, paddle_x, spin
        low = np.array([0, 0, -1, -1, 0, -5], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 5], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("BabyPong Bounce (Physical Collision)")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.ball_x = np.random.rand()
        self.ball_y = 0.5

        angle = random.uniform(-np.pi / 3, np.pi / 3)  # 擴大角度範圍
        speed = random.uniform(self.base_speed * 0.8, self.base_speed * 1.2)  # 隨機速度
        direction = random.choice([-1, 1])
        self.ball_vx = speed * np.sin(angle)
        self.ball_vy = speed * np.cos(angle) * direction

        # 加入隨機轉速
        self.spin = random.uniform(-10, 10)
        self.spin_angle = 0.0

        self.paddle_x = 0.5
        self.bounces = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            self.paddle_x,
            self.spin
        ], dtype=np.float32)

    def _scale_difficulty(self):
        scale_factor = 1 + (self.bounces // self.speed_scale_every) * self.speed_increment
        self.ball_vx *= scale_factor
        self.ball_vy *= scale_factor

    def step(self, action):
        reward = 0.0

        # 移動擋板的懲罰
        if action != 1:
            reward -= self.move_penalty

        # 零：左移，二：右移
        if action == 0:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed
        self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

        # Magnus effect (馬格努斯效應)
        if self.enable_spin:
            self.ball_vx += self.magnus_factor * self.spin * self.ball_vy

        # 更新球位置
        old_ball_x = self.ball_x
        old_ball_y = self.ball_y
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # X 邊界反彈
        if self.ball_x <= 0.0 or self.ball_x >= 1.0:
            self.ball_vx *= -1

        # Y 上邊界反彈
        if self.ball_y >= 1.0:
            self.ball_vy *= -1

        # --- 以下是「更物理的」底部擋板碰撞 ---
        # 判斷是否抵達底部 y <= 擋板上緣
        collision_y = (self.ball_y <= (self.paddle_height / self.window_size))
        if collision_y:
            # 先檢查 x 範圍是否碰到擋板
            paddle_min = self.paddle_x - (self.paddle_width / self.window_size) / 2
            paddle_max = self.paddle_x + (self.paddle_width / self.window_size) / 2

            if paddle_min <= self.ball_x <= paddle_max:
                # 真正撞到擋板，做更物理的處理
                # 1) 計算板子水平速度 (如需要可以偵測本 step 與上 step 的 paddle_x)
                #    這裡若想簡化，直接視板子不動 (u=0) 或依 action 來設定
                u = 0.0
                if action == 0:
                    # 往左移 => u 為負
                    u = -self.paddle_speed
                elif action == 2:
                    # 往右移 => u 為正
                    u = self.paddle_speed

                # 2) 法向速度: 下方板子 normal 向上 => vn = -ball_vy
                vn = -self.ball_vy
                # 3) 切向速度: 水平 => vt = ball_vx
                vt = self.ball_vx
                # 4) 角速度
                omega = self.spin

                # 5) 呼叫碰撞函式 (&#8203;:contentReference[oaicite:2]{index=2})
                vn_post, vt_post, omega_post = collide_sphere_with_moving_plane(
                    vn,
                    vt,
                    u,
                    omega,
                    self.restitution,
                    self.friction,
                    self.ball_mass,
                    self.world_ball_radius
                )

                # 更新球的速度與自轉
                self.ball_vy = -vn_post  # 因 vn_post 是以擋板 normal 為正方向
                self.ball_vx = vt_post
                self.spin = omega_post

                # 校正球的位置，避免「穿透」板子
                self.ball_y = (self.paddle_height / self.window_size)

                # 計算一些獎勵
                reward += self.bounce_reward
                # 完美擊中 (貼近擋板中線)
                if abs(self.ball_x - self.paddle_x) < 0.02:
                    reward += self.perfect_hit_bonus

                self.bounces += 1
                self._scale_difficulty()
            else:
                # 撞到底部但沒撞到擋板 (miss)
                reward -= self.miss_penalty
                return self._get_obs(), reward, True, False, {}

        # 中心懲罰 (不維持擋板在中間)
        center_penalty = abs(self.paddle_x - 0.5)
        reward -= self.center_penalty_weight * center_penalty

        done = self.bounces >= self.max_bounces
        return self._get_obs(), reward, done, False, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        cx = int(self.ball_x * self.window_size)
        cy = int((1 - self.ball_y) * self.window_size)
        paddle_px = int(self.paddle_x * self.window_size)

        # 畫球
        pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), self.ball_radius)

        # 球上加上一條簡單「spin線」做旋轉效果
        self.spin_angle += self.spin
        r = self.ball_radius - 2
        end_x = int(cx + r * np.cos(self.spin_angle))
        end_y = int(cy + r * np.sin(self.spin_angle))
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (end_x, end_y), 2)

        # 畫擋板 (綠色)
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (
                paddle_px - self.paddle_width // 2,
                self.window_size - self.paddle_height,
                self.paddle_width,
                self.paddle_height
            )
        )
        pygame.display.flip()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                quit()

    def close(self):
        pygame.quit()
