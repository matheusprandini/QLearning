class Agent():

    def __init__(self, name='Default', actions=None):
        self.name = name
        self.actions = actions
        self.position = None

    def get_number_actions(self):
        return len(self.actions)