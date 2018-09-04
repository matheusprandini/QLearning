import numpy as np

class GridWorld():

    def __init__(self, size, number_obstacles):
        self.size = size
        self.number_obstacles = number_obstacles
        self.environment = self.create_environment(size, number_obstacles)

    def create_environment(self, size, number_obstacles):
        env = np.zeros((size, size)) # Initialize environment with zeros
        env[size - 1][size - 1] = 1 # Goal position
        
		# Create obstacles in the environment
        obstacles_position = self.generate_obstacles(size, number_obstacles)

        for pos in obstacles_position:
            x = pos[0] # Get row index
            y = pos[1] # Get column index
            env[x][y] = -1 # Create obstacle
        
        return env

    def generate_obstacles(self, size, number_obstacles):
        positions = [] # Positions for the obstacles -> [x,y] (x -> row index; y -> column index)
        counter = 0 # Counter for obstacles generated
        initial_position = 0 # Initial position index
        goal_position = size - 1 # Goal position index
		
		# Generate "number_obstacles" of obstacles
        while counter < number_obstacles:
            x = np.random.choice(size, 1)[0] # Random x in [0, size -1] range (row index)
            y = np.random.choice(size, 1)[0] # Random y in [0, size -1] range (column index)
			
			# Append position if it's not the initial position nor goal position and if there's no other obstacle in the position
            if ((x != initial_position or y != initial_position) and (x != goal_position or y != goal_position)) and [x,y] not in positions:
                positions.append([x,y]) # Append pair (x,y)
                counter += 1 # Increment counter

        return positions
		
    def get_number_states(self):
        return self.size ** 2
		
    def execute_action(self, current_state, action):

        # Up
        if action == 0:
            new_state_position = (current_state[0] - 1, current_state[1])
            new_state_value = self.environment[current_state[0] - 1][current_state[1]]
        # Down
        elif action == 1:
            new_state_position = (current_state[0] + 1, current_state[1])
            new_state_value = self.environment[current_state[0] + 1][current_state[1]]
        # Left
        elif action == 2:
            new_state_position = (current_state[0], current_state[1] - 1)
            new_state_value = self.environment[current_state[0]][current_state[1] - 1]
        # Right
        elif action == 3:
            new_state_position = (current_state[0], current_state[1] + 1)
            new_state_value = self.environment[current_state[0]][current_state[1] + 1]
        # None
        else:
            new_state_position = (current_state[0], current_state[1])
            new_state_value = self.environment[current_state[0]][current_state[1]]

        new_state = list([new_state_position, new_state_value])

        return new_state