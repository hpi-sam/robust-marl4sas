class RankLearner:
    def __init__(self, stage, args):
        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of agents
        self.stage = stage
        self.args = args

    def _build_network(self):
        """ build the ranking network """





        # test failing of linter 123456789097643212345678909765432123456789098765432234567890987654323456789876543234567876543
        raise NotImplementedError

    def sort_actions(self, inputs):
        """ Sorts the actions of each agents """
        raise NotImplementedError

    def learn(self):
        """ learn to rank """
        raise NotImplementedError
