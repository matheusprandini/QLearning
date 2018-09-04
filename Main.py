from Agent import Agent
from GridWorld import GridWorld
from QLearning import QLearning
import numpy as np

actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'none': 4}

agent = Agent('Agent 1', actions)
gridWorld = GridWorld(4,3)
qLearning = QLearning(agent, gridWorld)

qLearning.training_q_learning()

print('Environment: ')
print(gridWorld.environment)

print('\nFinal Q-table:')
print(qLearning.q_table)

print('\nQ Learning Final Path:')
qLearning.execute_q_learning()