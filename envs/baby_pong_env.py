# baby_pong_env.py - 支援旋轉視覺化與球厚度碰撞處理

import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
import random

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
                 magnus_factor=0.01):
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
        self.paddle_height = 10  # 擋板厚度
        self.ball_radius = 10

        self.enable_spin = enable_spin
        self.magnus_factor = magnus_factor

        self._init_pygame()

        low = np.array([0, 0, -1, -1, 0, -5], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 5], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("BabyPong Bounce")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.ball_x = np.random.rand()
        self.ball_y = 0.5

        angle = random.uniform(-np.pi / 2.5, np.pi / 2.5)  # 擴大角度範圍
        speed = random.uniform(self.base_speed * 0.5, self.base_speed * 2.5)  # 隨機速度
        direction = random.choice([-1, 1])
        self.ball_vx = speed * np.sin(angle)
        self.ball_vy = speed * np.cos(angle) * direction

        self.spin = random.uniform(-3, 3)  # 加入隨機轉速
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
            self.spin  # 新增轉速
        ], dtype=np.float32)


    def _scale_difficulty(self):
        scale_factor = 1 + (self.bounces // self.speed_scale_every) * self.speed_increment
        self.ball_vx *= scale_factor
        self.ball_vy *= scale_factor

    def step(self, action):
        reward = 0.0

        if action != 1:
            reward -= self.move_penalty

        if action == 0:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed
        self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

        # Magnus effect
        if self.enable_spin:
            self.ball_vx += self.magnus_factor * self.spin * self.ball_vy

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x <= 0.0 or self.ball_x >= 1.0:
            self.ball_vx *= -1
        if self.ball_y >= 1.0:
            self.ball_vy *= -1

        # 球底部碰撞 + 考慮球半徑與板厚度
        collision_y = self.ball_y <= (self.paddle_height / self.window_size)
        if collision_y:
            paddle_min = self.paddle_x - (self.paddle_width / self.window_size) / 2
            paddle_max = self.paddle_x + (self.paddle_width / self.window_size) / 2
            if paddle_min <= self.ball_x <= paddle_max:
                self.ball_y = self.paddle_height / self.window_size  # 碰撞修正
                self.ball_vy *= -1
                reward += self.bounce_reward

                self.spin = (self.ball_x - self.paddle_x) * 20

                if abs(self.ball_x - self.paddle_x) < 0.02:
                    reward += self.perfect_hit_bonus

                self.bounces += 1
                self._scale_difficulty()
            else:
                reward -= self.miss_penalty
                return self._get_obs(), reward, True, False, {}

        center_penalty = abs(self.paddle_x - 0.5)
        reward -= self.center_penalty_weight * center_penalty

        done = self.bounces >= self.max_bounces
        return self._get_obs(), reward, done, False, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        cx = int(self.ball_x * self.window_size)
        cy = int((1 - self.ball_y) * self.window_size)
        paddle_px = int(self.paddle_x * self.window_size)

        pygame.draw.circle(self.screen, (255, 0, 0), (cx, cy), self.ball_radius)

        # 球紋路旋轉線條
        self.spin_angle += self.spin
        r = self.ball_radius - 2
        end_x = int(cx + r * np.cos(self.spin_angle))
        end_y = int(cy + r * np.sin(self.spin_angle))
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (end_x, end_y), 2)

        pygame.draw.rect(self.screen, (0, 255, 0), (paddle_px - self.paddle_width // 2,
                                                    self.window_size - self.paddle_height,
                                                    self.paddle_width, self.paddle_height))
        pygame.display.flip()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                quit()

    def close(self):
        pygame.quit()