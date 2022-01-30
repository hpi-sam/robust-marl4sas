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

        '''Create a new instance of the mRUBiS environment class'''
        self.external_start = external_start

        # Put your command line here (In Eclipse: Run -> Run Configurations... -> Show Command Line)
        with open(json_path, 'r') as f:
            variable_paths = json.load(f)

        self.launch_args = [
            variable_paths['java_path'],
            '-DFile.encoding=UTF-8',
            '-classpath',
            variable_paths['dependency_paths'],
            '-XX:+ShowCodeDetailsInExceptionMessages',
            'mRUBiS_Tasks.Task_1',
        ]

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
        print(actions)
        self.communicator.println(json.dumps(actions))
        raise NotImplementedError

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
        if self.observation is None:            
            print("Getting intitial state")
            self.observation = self._get_initial_state()
            print("Got initial state")
        self.prior_utility = self._get_current_utility()
        self.action_space = [components for shops, components in self.observation.items()][0].keys()
        return self.observation

    def render(self):
        """ Renders the environment. """
        raise NotImplementedError

    def close(self):
        """ Override close in your subclass to perform any necessary cleanup. """
        self._send_exit_message()
        self.communicator.close_socket()

        if not self.external_start:
            self._stop_mrubis()

    def seed(self):
        """ Sets the seed for this env's random number generator(s). """
        raise NotImplementedError

    def last(self):
        """ returns last state, reward, terminated, info """
        raise NotImplementedError

    def _start_mrubis(self):
        '''Launch mRUBiS as a subprocess. NOTE: Unstable. Manual startup from Eclipse is more robust.'''
        self.mrubis_process = Popen(
            self.launch_args,
            stdin=PIPE,
            stdout=PIPE,
            shell=False,
            cwd="../mRUBiS/ML_based_Control"
        )

    def _stop_mrubis(self):
        '''Terminate the mRUBiS process'''
        self.mrubis_process.terminate()

    # TODO: All methods here can probably be removed

    def _get_initial_state(self):
        '''Query mRUBiS for the number of shops, get their initial states'''
        # self.number_of_shops = self._get_from_mrubis(
        #     'get_number_of_shops').get('number_of_shops')
        # logger.info(f'Number of mRUBIS shops: {self.number_of_shops}')
        # for _ in range(self.number_of_shops):
        #     shop_state = self._get_from_mrubis('get_initial_state')
        #     shop_name = next(iter(shop_state))
        #     self.mrubis_state[shop_name] = shop_state[shop_name]
        # return self.communicator.get_from_mrubis('get_initial_state')
        return self.communicator.read_json()

    def _update_number_of_issues_in_run(self):
        '''Update the number of issues present in the current run'''
        self.number_of_issues_in_run = self._get_from_mrubis(
            'get_number_of_issues_in_run').get('number_of_issues_in_run')

    def _get_from_mrubis(self, message):
        """Send a message to mRUBiS and return the response as a dictionary"""
        self.communicator.println("message")
        logger.debug(f'Waiting for mRUBIS to answer to message {message}')
        return self.communicator.read_json()

    def _send_rule_to_execute(self, issue, rule):
        '''Send a rule to apply to an issue to mRUBiS'''
        shop_name, component_name, _, issue_name, _, _ = self._get_info_from_issue(
            issue)
        picked_rule_message = {shop_name: {issue_name: {component_name: rule}}}
        logger.info(
            f"{shop_name}: Handling {issue_name} on {component_name} with {rule}")
        logger.debug('Sending selected rule to mRUBIS...')
        self.communicator.println(json.dumps(picked_rule_message))
        logger.debug("Waiting for mRUBIS to answer with 'rule_received'...")
        data = self.communicator.readln()
        if data == 'rule_received':
            logger.debug('Rule transmitted successfully.')
        # Remember components that have been fixed in this run
        if self.components_fixed_in_this_run.get(shop_name) is None:
            self.components_fixed_in_this_run[shop_name] = []
        self.components_fixed_in_this_run[shop_name].append(
            (issue_name, component_name))

    def _send_order_in_which_to_apply_fixes(self, order_tuples):
        '''Send the order in which to apply the fixes to mRUBiS'''
        logger.debug('Sending order in which to apply fixes to mRUBIS...')
        order_dict = {idx: {
            'shop': fix_tuple[0],
            'issue': fix_tuple[1],
            'component': fix_tuple[2]
        } for idx, fix_tuple in enumerate(order_tuples)}
        '''
        for issueComponent in order_dict:
            self.socket.send(json.dumps(issueComponent))
            data = self.socket.recv(64000)
        '''
        self.communicator.println(json.dumps(order_dict))
        logger.debug(
            "Waiting for mRUBIS to answer with 'fix_order_received'...")
        data = self.communicator.readln()
        if data == 'fix_order_received':
            logger.debug('Order transmitted successfully.')

    def _send_exit_message(self):
        '''Tell mRUBiS to stop its main loop'''
        self.communicator.println("exit")
        _ = self.communicator.readln()
