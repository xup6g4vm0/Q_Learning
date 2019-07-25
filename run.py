from maze_env import Maze
from model import QLearning

def train():
  for episode in range(100):
    obs = env.reset()

    Reward = 0

    while True:
      # env.render()

      action = RL.choose_action(str(obs))

      ns, reward, done, _ = env.step(action)
      Reward += reward

      RL.learn(str(obs), action, reward, str(ns), done)

      obs = ns

      if done:
        print('episode: {}, Reward: {}'.format(episode, Reward))
        break

def _eval():
  for episode in range(10):
    obs = env.reset()

    Reward = 0

    while True:
      env.render()

      action = RL.choose_action(str(obs), True)

      obs, reward, done, _ = env.step(action)
      Reward += reward

      if done:
        print('Reward: {}'.format(Reward))
        break
      
if __name__ == '__main__':
  env = Maze()
  RL = QLearning(actions=list(range(env.n_actions)))

  train()

  _eval()
