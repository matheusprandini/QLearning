import numpy as np

class QLearning():

    def __init__(self, agent, environment, epsilon=1, epsilon_decay=0.001, gamma=0.9, learning_rate=0.1, number_episodes=1000):
        self.agent = agent
        self.environment = environment
        self.states = list(np.ndenumerate(environment.environment))
        self.number_states = environment.get_number_states()
        self.number_actions = agent.get_number_actions()
        self.q_table = self.initialize_q_table()
        self.reward_table, self.valid_actions_table, self.dict_states = self.initialize_reward_table(environment, agent.actions)
        self.number_episodes = number_episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
		
    def initialize_q_table(self):
        return np.zeros((self.number_states, self.number_actions))

    def initialize_reward_table(self, environment, actions):
        r_table = np.zeros((self.number_states, self.number_actions))
        valid_actions_table = np.array([])
        environment_states = self.states
        dict_states = {}
        counter_states = 0

        for state in environment_states:
		
            dict_states[state[0]] = counter_states
		
            valid_actions_state = np.array([])
		
		    # Getting row_index and column_index
            row_index = state[0][0]
            column_index = state[0][1]
			
            # Simulating actions in the state

			# Action "up"
            if row_index - 1 >= 0:

                new_state = environment.environment[row_index - 1][column_index]
				
                r_table[counter_states][actions['up']] = self.get_reward_state(new_state)
                valid_actions_state = np.append(valid_actions_state, actions['up'])
			
            else:

                r_table[counter_states][actions['up']] = -np.inf
                self.q_table[counter_states][actions['up']] = -np.inf			

            # Action "down"
            if row_index + 1 <= environment.size - 1:
			
                new_state = environment.environment[row_index + 1][column_index]
				
                r_table[counter_states][actions['down']] = self.get_reward_state(new_state)
                valid_actions_state = np.append(valid_actions_state, actions['down'])
				
            else:
			
                r_table[counter_states][actions['down']] = -np.inf
                self.q_table[counter_states][actions['down']] = -np.inf	
			
            # Action "left"			
            if column_index - 1 >= 0:

                new_state = environment.environment[row_index][column_index - 1]
				
                r_table[counter_states][actions['left']] = self.get_reward_state(new_state)
                valid_actions_state = np.append(valid_actions_state, actions['left'])
				
            else:
			
                r_table[counter_states][actions['left']] = -np.inf
                self.q_table[counter_states][actions['left']] = -np.inf	
				
            # Action "right"
            if column_index + 1 <= environment.size - 1:
            
                new_state = environment.environment[row_index][column_index + 1]

                r_table[counter_states][actions['right']] = self.get_reward_state(new_state)
                valid_actions_state = np.append(valid_actions_state, actions['right'])
				
            else:
			
                r_table[counter_states][actions['right']] = -np.inf
                self.q_table[counter_states][actions['right']] = -np.inf	
				
            # Action "none"
            new_state = environment.environment[row_index][column_index]
			
            r_table[counter_states][actions['none']] = self.get_reward_state(new_state)
            valid_actions_state = np.append(valid_actions_state, actions['none'])

            for i in range(self.number_actions - len(valid_actions_state)):
                valid_actions_state = np.concatenate((valid_actions_state, [-1]), axis=0)

            if counter_states == 0:
                valid_actions_table = [valid_actions_state]
            else:
                valid_actions_table = np.concatenate((valid_actions_table, [valid_actions_state]), axis=0)
				
            counter_states += 1
			
        return r_table, valid_actions_table, dict_states
	
    def get_reward_state(self, new_state):
        if new_state == 1:
            return 10
        elif new_state == -1:
            return -10
        else:
            return -1

    def execute_episode_with_random_actions(self):
	
        # Select a starting state
        initial_state = self.states[0]
        current_state = initial_state
        self.agent.position = current_state[0]
        current_state_number = self.dict_states[self.agent.position]

		# Until the end of the episode
        while self.agent.position != self.states[-1][0]:
            
            # Get the valid actions from the current state
            valid_actions = self.valid_actions_table[current_state_number].astype(int)			
			
			# Select an action among all the possible actions
            action = -1

            while action == -1:
                action = np.random.choice(valid_actions)
			
            # Execute the action and reach the new_state
            new_state = self.environment.execute_action(current_state[0], action)
            new_state_number = self.dict_states[new_state[0]]

            # Get the immediate reward
            immediate_reward = self.get_reward_state(new_state[1])
            
			# Verify the best action on the new_state (max future reward - max q_value from new_state)
            best_future_reward = np.amax(self.q_table[new_state_number])
			
            # Update Q-Table
            self.q_table[current_state_number][action] = ((1 - self.learning_rate) * self.q_table[current_state_number][action]) + (self.learning_rate * (immediate_reward + (self.gamma * best_future_reward)))

			# Update the current state
            current_state = new_state
            current_state_number = new_state_number
            self.agent.position = current_state[0]
			
    def execute_episode_with_epsilon_greedy(self):
	
        # Select a starting state
        initial_state = self.states[0]
        current_state = initial_state
        self.agent.position = current_state[0]
        current_state_number = self.dict_states[self.agent.position]

		# Until the end of the episode
        while self.agent.position != self.states[-1][0]:
                
            # Get the valid actions from the current state
            valid_actions = self.valid_actions_table[current_state_number].astype(int)			
			
			# Select an action among all the possible actions using epsilon-greedy strategy (Exploration x Exploitation)
            action = -1
				
			# Generate a random float number between 0 and 1
            random_number = np.random.uniform(0,1)
			
			# Exploration -> random choice (action)
            if random_number <= self.epsilon:
                while action == -1:
                    action = np.random.choice(valid_actions)
            # Exploitation -> best action based on q_value
            else:
                action = self.q_table[current_state_number].argmax(axis=0)
			
            # Execute the action and reach the new_state
            new_state = self.environment.execute_action(current_state[0], action)
            new_state_number = self.dict_states[new_state[0]]

            # Get the immediate reward
            immediate_reward = self.get_reward_state(new_state[1])
            
			# Verify the best action on the new_state (max future reward - max q_value from new_state)
            best_future_reward = np.amax(self.q_table[new_state_number])
			
            # Update Q-Table
            self.q_table[current_state_number][action] = ((1 - self.learning_rate) * self.q_table[current_state_number][action]) + (self.learning_rate * (immediate_reward + (self.gamma * best_future_reward)))

			# Update the current state
            current_state = new_state
            current_state_number = new_state_number
            self.agent.position = current_state[0]
            
    def training_q_learning(self, mode=1):

        for i in range(self.number_episodes):

            if mode == 0:
                self.execute_episode_with_random_actions()
            else:
                self.execute_episode_with_epsilon_greedy()
                if self.epsilon > 0.1:
                    self.epsilon -= self.epsilon_decay
					
    def execute_q_learning(self):
        
		# Select a starting state
        initial_state = self.states[0]
        current_state = initial_state
        self.agent.position = current_state[0]
        current_state_number = self.dict_states[self.agent.position]
        total_reward = 0

		# Until the end of the episode
        while self.agent.position != self.states[-1][0]:
		
            print(str(current_state[0]) + ' - Sum Rewards: ' + str(total_reward))
			
			# Get the action with max q_value
            action = self.q_table[current_state_number].argmax(axis=0)
			
            # Execute the action and reach the new_state
            new_state = self.environment.execute_action(current_state[0], action)
            new_state_number = self.dict_states[new_state[0]]

			# Update the current state and total reward
            total_reward += self.q_table[current_state_number][action]
            current_state = new_state
            current_state_number = new_state_number
            self.agent.position = current_state[0]
        
        print(str(current_state[0]) + ' - Total Reward: ' + str(total_reward))