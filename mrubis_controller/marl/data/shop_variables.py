import random

shop_name = "mRUBiS #"

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


def get_failing_dependencies(seed=1):
    """ Returns for each component index a sorted list of most probable component indices
        which would fail in the observations.
        This should simulate the shop internal component dependencies.
    """
    random.seed(1)
    dependencies = []
    for i in range(18):
        probabilities = {i: random.uniform(0, 1) for i in range(18)}
        dependencies.append(sorted(probabilities, key=probabilities.get, reverse=True))
    return dependencies
