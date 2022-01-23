import random
from shop_variables import shop_name, component_names, get_failing_dependencies


def _generate_shop(failing_component, failure_name="CF2"):
    """ returns a shop with only currently necessary attributes per component """
    shop = {}
    shop_utility = 0
    for component in component_names:
        shop[component] = {
            "failure_name": failure_name if component in failing_component else "None",
            "component_utility": random.uniform(3000, 25000)
        }
        shop_utility += shop[component]["component_utility"]
    for component in component_names:
        shop[component]["shop_utility"] = shop_utility

    return shop


def _generate_state_fails(number_of_shops):
    """ returns actual failing component indices """
    failing_components = []
    for i in range(number_of_shops):
        failing_components.append(random.randint(1, 18))
    return failing_components


def _get_offset_fails(failing_component_index, number_of_failures, offset):
    """ returns a list of false failing components starting with an offset of the actual failing component """
    return [component_names[(failing_component_index + offset + j) % 18] for j in
            range(number_of_failures)]


def _get_dependency_fails(failing_component_dependencies, number_of_failures, offset):
    """ returns most probable false failing components
        the offset adds a randomness factor
    """
    failing_component_indices = random.sample(failing_component_dependencies[:(number_of_failures + offset)],
                                              number_of_failures)
    return [component_names[index] for index in failing_component_indices]


def generate_shops_with_offset_failures(number_of_shops, number_of_failures=1, offset=1, use_dependencies=True, seed=1):
    """ returns shop observations and the actual failing component per shop """
    failing_dependencies = get_failing_dependencies(1)
    random.seed(seed)
    shops_obs = {}
    shops_state = {}
    failing_component_indices = _generate_state_fails(number_of_shops)
    for i in range(1, 1 + number_of_shops):
        if use_dependencies:
            obs_fails = _get_dependency_fails(failing_dependencies[failing_component_indices[i - 1]],
                                              number_of_failures, offset)
        else:
            obs_fails = _get_offset_fails(failing_component_indices[i - 1], number_of_failures, offset)
        shops_obs[shop_name + str(i)] = _generate_shop(obs_fails)
        shops_state[shop_name + str(i)] = component_names[failing_component_indices[i - 1]]

    return shops_obs, shops_state


if __name__ == "__main__":
    generate_shops_with_offset_failures(number_of_shops=2, number_of_failures=3, offset=2)
