from numpy.testing import assert_array_equal

from mrubis_controller.marl.agent_3 import Agent3 as Agent


def test_save_and_load_models():
    action_space = ['Availability Item Filter', 'Bid and Buy Service', 'Seller Reputation Item Filter',
                    'User Management Service', 'Reputation Service', 'Comment Item Filter',
                    'Recommendation Item Filter', 'Authentication Service', 'Persistence Service',
                    'Item Management Service', 'Buy Now Item Filter', 'Query Service', 'Past Sales Item Filter',
                    'Inventory Service', 'Region Item Filter', 'Category Item Filter', 'Last Second Sales Item Filter',
                    'Future Sales Item Filter']
    shops = {'mRUBiS #1'}
    episode = 0
    agent = Agent(0, shops, action_space, None)
    agent.save(episode)
    load_model = {"start_time": agent.start_time, "episode": episode}
    actor, critic, policy = agent.load_models(load_model)

    # loaded and saved have equal weights and bias
    assert_array_equal(actor.get_layer('dense1').get_weights()[0], agent.actor.get_layer('dense1').get_weights()[0])
    assert_array_equal(actor.get_layer('dense1').get_weights()[1], agent.actor.get_layer('dense1').get_weights()[1])
    assert_array_equal(critic.get_layer('dense1').get_weights()[0], agent.critic.get_layer('dense1').get_weights()[0])
    assert_array_equal(critic.get_layer('dense1').get_weights()[1], agent.critic.get_layer('dense1').get_weights()[1])

    # loaded actor and critic share the same layer
    assert actor.get_layer('dense1') == critic.get_layer('dense1')
