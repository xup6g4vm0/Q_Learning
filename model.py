import numpy as np
import pandas as pd

class QLearning:
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    self.actions = actions
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epilson = e_greedy
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

  def choose_action(self, state, _eval=False):
    self.check_state_exist(state)

    if _eval or np.random.rand() < self.epilson:
      state_action = self.q_table.loc[state, :]

      action = np.random.choice(state_action[state_action == np.max(state_action)].index)
    else:
      action = np.random.choice(self.actions)

    return action

  def learn(self, s, a, r, ns, done):
    self.check_state_exist(ns)
    q_predict = self.q_table.loc[s,a]
    if not done:
      q_target = r + self.gamma * self.q_table.loc[ns, :].max()
    else:
      q_target = r

    self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      self.q_table = self.q_table.append(
        pd.Series(
          [0]*len(self.actions),
          index=self.q_table.columns,
          name=state
        )
      )
