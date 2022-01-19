def prepare_step(inputs):
    index = 0
    sorted_actions = {}
    for sublist in inputs:
        for item in sublist:
            sorted_actions[index] = item
            index += 1
    return sorted_actions


class RankLearner:
    def __init__(self, stage, args):
        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of agents
        self.stage = stage
        self.args = args

    def _build_network(self):
        """ build the ranking network """
        raise NotImplementedError

    def sort_actions(self, inputs):
        """ Sorts the actions of each agents """
        if self.stage == 0:
            return prepare_step(inputs)
        else:
            raise NotImplementedError

    def learn(self):
        """ learn to rank """
        raise NotImplementedError

    def save(self, episode):
        """ save model """
        if self.stage != 0:
            raise NotImplementedError
