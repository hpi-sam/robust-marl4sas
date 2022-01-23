import random

component_names = ["Availability Item Filter",
                   "Bid and Buy Service",
                   "Seller Reputation Item Filter",
                   "User Management Service",
                   "Reputation Service",
                   "Comment Item Filter",
                   "Recommendation Item Filter",
                   "Authentication Service",
                   "Persistence Service",
                   "Item Management Service",
                   "Buy Now Item Filter",
                   "Query Service",
                   "Past Sales Item Filter",
                   "Inventory Service",
                   "Region Item Filter",
                   "Category Item Filter",
                   "Last Second Sales Item Filter",
                   "Future Sales Item Filter"]
shop_name = "mRUBiS #"


def _generate_shop(failing_component, failure_name="CF2"):
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
    failing_components = []
    for i in range(number_of_shops):
        failing_components.append(random.randint(1, 18))
    return failing_components


def generate_shops_with_offset(number_of_shops, offset=1, seed=1):
    random.seed(seed)
    shops_obs = {}
    shops_state = {}
    failing_component_indices = _generate_state_fails(number_of_shops)
    for i in range(1, 1 + number_of_shops):
        obs_fails = [component_names[(failing_component_indices[i - 1] + offset) % 18]]
        shops_obs[shop_name + str(i)] = _generate_shop(obs_fails)
        shops_state[shop_name + str(i)] = component_names[failing_component_indices[i - 1]]

    return shops_obs, shops_state


def generate_shops_with_multiple_failures(number_of_shops, number_of_failures=2, seed=1):
    random.seed(seed)
    shops_obs = {}
    shops_state = {}
    failing_component_indices = _generate_state_fails(number_of_shops)
    for i in range(1, 1 + number_of_shops):
        obs_fails = [component_names[i] for i in random.sample(range(0, 17), number_of_failures)]
        shops_obs[shop_name + str(i)] = _generate_shop(obs_fails)
        shops_state[shop_name + str(i)] = component_names[failing_component_indices[i - 1]]

    return shops_obs, shops_state


if __name__ == "__main__":
    generate_shops_with_offset(5)
