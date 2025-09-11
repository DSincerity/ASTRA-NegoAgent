import pulp
import math
from utils import create_capped_window

# Function to perform the actual Linear Programming for a single set of parameters
def solve_lp(max_point, lambda_factor, agents_value, partner_value, epsilon=0.0001):
    """
    Solves a Linear Programming problem for the given parameters.

    Args:
        max_point (int): Maximum points the agent can get.
        lambda_factor (float): Lambda factor balancing both parties' objectives.
        agents_value (dict): Agent's values for food, water, and firewood.
        partner_value (dict): Partner's values for food, water, and firewood.
        epsilon (float): Small adjustment to avoid floating point errors.

    Returns:
        tuple: A tuple containing the calculated scores and item allocations.
    """

    A_Val_Food = agents_value['food']
    A_Val_Water = agents_value['water']
    A_Val_Firewood = agents_value['firewood']

    # Agent's values adjusted to avoid floating point errors
    B_Val_Food = partner_value['food'] - epsilon
    B_Val_Water = partner_value['water'] - epsilon
    B_Val_Firewood = partner_value['firewood'] - epsilon

    # Define the LP problem
    problem = pulp.LpProblem("Negotiation_Strategy_Max_Points", pulp.LpMaximize)

    # Define variables
    X = pulp.LpVariable("X", 0, 3, cat='Integer')  # Food packages you get
    Y = pulp.LpVariable("Y", 0, 3, cat='Integer')  # Water packages you get
    Z = pulp.LpVariable("Z", 0, 3, cat='Integer')  # Firewood packages you get

    # Parameters
    B_threshold = 5  # Minimum points partner must get
    A_threshold = 10  # Minimum points agent must get

    # Objective function
    objective = (
        (A_Val_Food * X + A_Val_Water * Y + A_Val_Firewood * Z) +
        (1 - lambda_factor) * ((B_Val_Food * (3 - X) + B_Val_Water * (3 - Y) + B_Val_Firewood * (3 - Z)))
    )
    problem += objective

    # Constraints
    problem += A_Val_Food * X + A_Val_Water * Y + A_Val_Firewood * Z <= max_point, "MaxPointsYouGet"
    problem += A_Val_Food * X + A_Val_Water * Y + A_Val_Firewood * Z >= A_threshold, "AgentThreshold"
    problem += B_Val_Food * (3 - X) + B_Val_Water * (3 - Y) + B_Val_Firewood * (3 - Z) >= B_threshold, "PartnerThreshold"

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30, gapRel=0.01)
    problem.solve(solver)

    # Calculate scores
    A_score = A_Val_Food * X.varValue + A_Val_Water * Y.varValue + A_Val_Firewood * Z.varValue
    B_score = B_Val_Food * (3 - X.varValue) + B_Val_Water * (3 - Y.varValue) + B_Val_Firewood * (3 - Z.varValue)

    # Return results as a tuple
    return math.ceil(A_score), int(X.varValue), int(Y.varValue), int(Z.varValue)

# Function to loop over possible parameters and execute LP
def calculateBestOfferFromLP(maximum_value, lambda_value, agents_value, partner_value):
    """
    Iterates over parameters and calls the LP solver to find the best offers.

    Args:
        maximum_value (int): Maximum allowed value for the agent.
        lambda_value (float): Lambda value for balancing the objective.
        agents_value (dict): Agent's values for food, water, and firewood.
        partner_value (dict): Partner's values for food, water, and firewood.

    Returns:
        set: A set of tuples containing calculated scores and allocations.
    """
    set_of_offer = set()
    search_lower_bound = 5
    epsilon = 0.0001

    # Validate lambda value
    assert lambda_value is None or 0 <= lambda_value <= 1, "Lambda value should be between 0 and 1"

    # Create lambda lists for balancing
    lambda_lists = create_capped_window(lambda_value, window_size=0.3) if lambda_value else create_capped_window(0.5, window_size=0.5)

    # Loop through max values and lambda factors
    for mx in list(range(maximum_value, search_lower_bound, -1)):
        for l in lambda_lists:
            result = solve_lp(mx, l, agents_value, partner_value, epsilon)
            set_of_offer.add(result)

    return set_of_offer

if __name__ == "__main__":
    maximum_val = 30
    lambda_val = 0.3
    agents_preference_value = {'food': 5, 'water': 4, 'firewood': 3}
    partner_preference_value = {'food': 3, 'water': 4, 'firewood': 5}
    #partner_value = {'food': 5, 'water': 4, 'firewood': 3}
    lp_results1=calculateBestOfferFromLP(maximum_val, lambda_val, agents_preference_value, partner_preference_value)
    print(sorted(lp_results1, key=lambda x: x[0], reverse=True))
