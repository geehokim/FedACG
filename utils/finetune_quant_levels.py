
import itertools

import numpy as np
import math
from scipy import special

def refined_search(selected_alpha, refined_step, max_value, refinement_range=0.05):
    min_val = float('inf')

    ranges = [(max(refined_step, a - refinement_range), min(max_value, a + refinement_range)) for a in selected_alpha]
    values_list = [np.arange(start, end + refined_step, refined_step) for start, end in ranges]

    # Generate all combinations of alpha values
    alpha_combinations = itertools.product(*values_list)

    for k, alpha in enumerate(alpha_combinations):
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = np.array(np.meshgrid(*[[-1, 1]] * len(alpha))).T.reshape(-1, len(alpha))

        # Compute q for each b_combination
        q_values = np.array([np.sum(np.array(alpha) * b) for b in b_combinations])

        # Filter unique positive q values
        unique_positive_q = np.array(sorted(set(q_values[q_values > 0])))

        q_minus = unique_positive_q[:-1] - unique_positive_q[1:]
        q_plus = unique_positive_q[:-1] + unique_positive_q[1:]
        s = q_plus * 0.5
        q_square_diff = q_minus * q_plus

        total_error = np.sqrt(2. / np.pi) * (np.sum(q_minus * np.exp(-0.5 * (s ** 2))) - unique_positive_q[0]) \
                        + 0.5 * (np.sum(q_square_diff * special.erf(s / np.sqrt(2))) + (unique_positive_q[-1] ** 2 + 1))

        if min_val > total_error:
            min_val = total_error
            selected_alpha = alpha

        if k % 10000 == 0:
            print(f"{k} iterations | Alpha: {selected_alpha} | Min Err.: {min_val}")

    print(f"Selected | Alpha: {selected_alpha} | Min Err.: {min_val}")

# Example usage
selected_alpha = (0.05, 0.1, 0.2, 0.3375, 0.525, 0.9875, 1.3875, 1.45)
refined_step = 0.0375/9  # Step size for refined search
max_value = 1.8  # Maximum value for alpha
refinement_range = 0.0375/3  # Range around found alpha values for refinement

refined_search(selected_alpha, refined_step, max_value, refinement_range)
