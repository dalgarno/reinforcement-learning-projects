class Agent(object):
    def __init__(self, env, alpha=0.01, eps=0.2, gamma=0.98):
        super().__init__()
        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = None
