import gym
import numpy as np


class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=8,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.reward_threshold = self.env.spec.reward_threshold
    
    def reset(self):
        self.av_r = self.reward_memory()
        
        # Reset the original environment.
        img_rgb, info = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.stack_frames
        return np.array(self.stack), info
    
    def step(self, action):
        total_reward = 0
        for i in range(self.skip_frames):
            img_rgb, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            done = True if self.av_r(reward) <= -0.1 else False
            if done or terminated or truncated:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.stack_frames
        return np.array(self.stack), total_reward, terminated, truncated, done, info
    
    def render(self, *arg):
        self.env.render(*arg)
    
    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return gray
    
    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)
        
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        
        return memory
        