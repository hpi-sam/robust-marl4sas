import gym
import logging

import json
from time import sleep
from subprocess import PIPE, Popen

from chunkedsocketcommunicator import ChunkedSocketCommunicator

logging.basicConfig()
logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)


class MrubisEnv(gym.Env):
    def __init__(self, json_path='path.json', external_start=True):
        super(MrubisEnv, self).__init__()
        self.action_space = None
        self.observation_space = None
        self.communicator = None
        self.prior_utility = None  # used for calculation of diff as a reward
        self.t = 0
        self.termination_t = 3
        self.inner_t = 0
        self.stats = {}

        '''Create a new instance of the mRUBiS environment class'''
        self.external_start = external_start

        # Put your command line here (In Eclipse: Run -> Run Configurations... -> Show Command Line)
        # with open(json_path, 'r') as f:
        #     variable_paths = json.load(f)

        '''
        self.launch_args = [
            variable_paths['java_path'],
            '-DFile.encoding=UTF-8',
            '-classpath',
            variable_paths['dependency_paths'],
            '-XX:+ShowCodeDetailsInExceptionMessages',
            'mRUBiS_Tasks.Task_1',
        ]
        '''

        self.run_counter = 0
        self.number_of_shops = 0
        self.number_of_issues_per_shop = {}
        self.current_issue = {}
        self.mrubis_state = {}
        self.mrubis_state_history = []
        self.fix_history = []
        self.current_fixes = None
        self.mrubis_process = None
        self.communicator = None
        self.observation = None

        # initial state and action
        self.reset()

    def _get_current_utility(self):
        return {shop: float(list(components.items())[0][1]['shop_utility']) for shop, components in
                self.observation.items()}

    def step(self, actions):
        """ Returns observation, reward, terminated, info """
        # TODO: Send actions to mRUBiS
        # TODO: Get observation from mRUBiS
        # TODO: Determine reward (calculate from observation)
        # TODO: What to do once failure propagation is implemented?
        self.inner_t += 1
        if actions is not None:
            self.communicator.println(json.dumps(actions))
        else:
            self.communicator.println(json.dumps([[]]))
        message = self.communicator.readln()
        assert message == "received"
        self.observation = self._get_state()

        _reward = self._get_reward()

        for shop in _reward[0]:
            if _reward[0][shop] > 0:
                self.stats[shop] = self.inner_t

        if actions is None or self._is_fixed():
            self.t += 1
            self.inner_t = 0

        return _reward, self.observation, self._terminated(), self._info()

    def reset(self):
        """ Returns initial observations and states """
        # TODO: send initiation data to mRUBiS (# shops, etc.) (for later!)
        # TODO: add additional argument for that
        # TODO: get initial observation from mRUBiS
        if self.communicator is None:
            self.communicator = ChunkedSocketCommunicator()
        # self.current_state = 0
        # self.current_state_name = list(self.observation_space_names[self.current_state])
        # self.last_action = None
        # self.last_action_name = None
        # self.successful_action = None
        # self.steps = 0

        if not self.external_start:
            self._start_mrubis()
            if self.mrubis_process.poll() is None:
                logger.info('MRUBIS is running')

        # Account for Java being slow to start on some systems
        sleep(0.5)

        self.t = 0
        self._reset_mrubis()
        self.observation = self._get_state()
        self.prior_utility = self._get_current_utility()
        self.action_space = [components for shops, components in self.observation.items()][0].keys()
        return self.observation

    def render(self):
        """ Renders the environment. """
        raise NotImplementedError

    def close(self):
        """ Override close in your subclass to perform any necessary cleanup. """
        self.communicator.send_exit_message()
        self.communicator.close_socket()

        if not self.external_start:
            self._stop_mrubis()

    def seed(self):
        """ Sets the seed for this env's random number generator(s). """
        raise NotImplementedError

    def last(self):
        """ returns last state, reward, terminated, info """
        raise NotImplementedError

    def _is_fixed(self):
        for shop in self.observation:
            for component in self.observation[shop]:
                if self.observation[shop][component]["failure_name"] != "None":
                    return False
        return True

    def _get_reward(self):
        """ returns the extracted reward per shop
        """
        current_utility = self._get_current_utility()
        diff_utility = {shop: current_utility[shop] - utility for shop, utility in self.prior_utility.items()}
        self.prior_utility = current_utility
        system_utility = sum(diff_utility.values())
        return diff_utility, system_utility

    def _terminated(self):
        return self.t == self.termination_t

    def _start_mrubis(self):
        '''Launch mRUBiS as a subprocess. NOTE: Unstable. Manual startup from Eclipse is more robust.'''
        self.mrubis_process = Popen(
            self.launch_args,
            stdin=PIPE,
            stdout=PIPE,
            shell=False,
            cwd="../mRUBiS/ML_based_Control"
        )

    def _reset_mrubis(self):
        self.communicator.println("reset")
        response = self.communicator.readln()
        if response == "resetting":
            return True
        else:
            return False

    def _stop_mrubis(self):
        '''Terminate the mRUBiS process'''
        self.mrubis_process.terminate()
        
    def _info(self):
        return {'t': self.t, 'stats': self.stats}

    def _get_state(self):
        '''Query mRUBiS for the state'''
        return self.communicator.get_from_mrubis("get_state")
