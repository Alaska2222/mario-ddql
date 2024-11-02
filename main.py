import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from agent import Agent
from OpenAIWrappers import NoopResetEnv, MaxAndSkipEnv,EpisodicLifeMario,WarpFrame,ScaledFloatFrame, FrameStack

if __name__ == "__main__":

   # Initialize enviroment
   env = gym_super_mario_bros.make("SuperMarioBros2-v1")
   env = JoypadSpace(env, SIMPLE_MOVEMENT)

   # Wrapping enviroment
"""
Code from OpenAI baseline
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

   env = NoopResetEnv(env, noop_max=30)
   env = MaxAndSkipEnv(env, skip=4)
   env = EpisodicLifeMario(env)
   env = WarpFrame(env)
   env = ScaledFloatFrame(env)
   env = FrameStack(env, 4)

   # Initialize Agent(Mario)
   agent = Agent(env)

   # Train Agent
   agent.train_model()

   # Save Agent
   agent.save_model()
