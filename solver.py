import pulp
import math

# Mappers for converting between class labels and integer values
# Support both original case and lowercase keys
self_class_to_int_mapper = {
    'price': {
        "1400": 4,
        "1200": 3,
        "1000": 2,
        "800": 1
    },
    'certification': {
        "None": 4,
        "none": 4,
        "Basic": 3,
        "basic": 3,
        "3rd-party": 2,
        "Full": 1,
        "full": 1
    },
    'payment': {
        "Full": 4,
        "full": 4,
        "1M": 3,
        "1m": 3,
        "3M": 2,
        "3m": 2,
        "6M": 1,
        "6m": 1
    }
}

# Reversed mapper: Always use canonical (uppercase/standard) keys
self_reversed_class_to_int_mapper = {
    'price': {
        4: "1400",
        3: "1200",
        2: "1000",
        1: "800"
    },
    'certification': {
        4: "None",    # Use uppercase None (canonical)
        3: "Basic",   # Use uppercase Basic (canonical)
        2: "3rd-party",
        1: "Full"     # Use uppercase Full (canonical)
    },
    'payment': {
        4: "Full",    # Use uppercase Full (canonical)
        3: "1M",      # Use uppercase M (canonical)
        2: "3M",
        1: "6M"
    }
}

partner_preference_value_to_int_mapper = {
    'price': {
        "1400": 1,
        "1200": 2,
        "1000": 3,
        "800": 4
    },
    'certification': {
        "None": 1,
        "none": 1,
        "Basic": 2,
        "basic": 2,
        "3rd-party": 3,
        "Full": 4,
        "full": 4
    },
    'payment': {
        "Full": 1,
        "full": 1,
        "1M": 2,
        "1m": 2,
        "3M": 3,
        "3m": 3,
        "6M": 4,
        "6m": 4
    }
}

partner_preference_value_to_int_mapper_reversed = {
    'price': {v: k for k, v in partner_preference_value_to_int_mapper['price'].items()},
    'certification': {v: k for k, v in partner_preference_value_to_int_mapper['certification'].items()},
    'payment': {v: k for k, v in partner_preference_value_to_int_mapper['payment'].items()}
}

def create_capped_window(center_value: float, window_size: float) -> list:
    """
    Create a list of values around center_value within window_size, capped between 0 and 1.
    """
    values = []
    step = window_size / 10
    start = max(0, center_value - window_size / 2)
    end = min(1, center_value + window_size / 2)

    current = start
    while current <= end:
        values.append(round(current, 4))
        current += step

    if center_value not in values:
        values.append(center_value)

    return sorted(list(set(values)))

# Function to perform the actual Linear Programming for a single set of parameters
def solve_lp(max_point, lambda_factor, agents_value, partner_value, epsilon=0.0001):
    """
    Solves a Linear Programming problem for the given parameters.

    Args:
        max_point (int): Maximum points the agent can get.
        lambda_factor (float): Lambda factor balancing both parties' objectives.
        agents_value (dict): Agent's values for price, certification, and payment.
        partner_value (dict): Partner's values for price, certification, and payment.
        epsilon (float): Small adjustment to avoid floating point errors.

    Returns:
        tuple: A tuple containing the calculated scores and item allocations.
    """

    A_Val_price = agents_value['price']
    A_Val_certification = agents_value['certification']
    A_Val_payment = agents_value['payment']

    # Agent's values adjusted to avoid floating point errors
    B_Val_price = partner_value['price'] - epsilon
    B_Val_certification = partner_value['certification'] - epsilon
    B_Val_payment = partner_value['payment'] - epsilon

    # Define the LP problem
    problem = pulp.LpProblem("Negotiation_Strategy_Max_Points", pulp.LpMaximize)

    # Define variables
    X = pulp.LpVariable("X", 0, 4, cat='Integer')  # price you get
    Y = pulp.LpVariable("Y", 0, 4, cat='Integer')  # certification you get
    Z = pulp.LpVariable("Z", 0, 4, cat='Integer')  # payment you get

    # Parameters
    B_threshold = 5  # Minimum points partner must get
    A_threshold = 10  # Minimum points agent must get

    # Objective function
    objective = (
        (A_Val_price * X + A_Val_certification * Y + A_Val_payment * Z) +
        (1 - lambda_factor) * ((B_Val_price * (5 - X) + B_Val_certification * (5 - Y) + B_Val_payment * (5 - Z)))
    )
    problem += objective

    # Constraints
    problem += A_Val_price * X + A_Val_certification * Y + A_Val_payment * Z <= max_point, "MaxPointsYouGet"
    problem += A_Val_price * X + A_Val_certification * Y + A_Val_payment * Z >= A_threshold, "AgentThreshold"
    problem += B_Val_price * (5 - X) + B_Val_certification * (5 - Y) + B_Val_payment * (5 - Z) >= B_threshold, "PartnerThreshold"

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30, gapRel=0.01)
    problem.solve(solver)

    # Calculate scores
    A_score = A_Val_price * X.varValue + A_Val_certification * Y.varValue + A_Val_payment * Z.varValue
    B_score = B_Val_price * (3 - X.varValue) + B_Val_certification * (3 - Y.varValue) + B_Val_payment * (3 - Z.varValue)

    # Return results as a tuple
    return math.ceil(A_score), int(X.varValue), int(Y.varValue), int(Z.varValue)

# Function to loop over possible parameters and execute LP
def calculateBestOfferFromLP(maximum_value, lambda_value, agents_value, partner_value):
    """
    Iterates over parameters and calls the LP solver to find the best offers.

    Args:
        maximum_value (int): Maximum allowed value for the agent.
        lambda_value (float): Lambda value for balancing the objective.
        agents_value (dict): Agent's values for price, certification, and payment.
        partner_value (dict): Partner's values for price, certification, and payment.

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
    agents_preference_value = {'price': 5, 'certification': 4, 'payment': 3}
    partner_preference_value = {'price': 3, 'certification': 4, 'payment': 5}
    lp_results1 = calculateBestOfferFromLP(maximum_val, lambda_val, agents_preference_value, partner_preference_value)
    print(sorted(lp_results1, key=lambda x: x[0], reverse=True))
