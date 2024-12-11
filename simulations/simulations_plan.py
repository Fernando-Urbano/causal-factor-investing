import pytest
import numpy as np
import os
import sys
import itertools
import pandas as pd
import portalocker

# Adjust the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from scenarios.causal_scenarios import (
    BackdoorAdjustmentScenario,
    InstrumentalVariableScenario
)

def keep_simulations():
    with open("simulations/keep_simulations.txt", "r") as f:
        keep_simulations = f.read().splitlines()
    return keep_simulations[0] == "True"

def read_and_delete_second_line(file_path):
    """
    Reads and deletes the second line of a text file using portalocker for file locking.
    Waits until the file is available if it is locked by another process.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: The second line of the file, or None if the file has fewer than two lines.
    """
    try:
        with portalocker.Lock(file_path, 'r+', timeout=None) as file:
            # Read all lines
            lines = file.readlines()
            
            # If the file has fewer than two lines, return None
            if len(lines) < 2:
                return None
            
            # Get the second line
            second_line = lines[1].strip()
            
            # Write back all lines except the second
            file.seek(0)  # Move to the start of the file
            file.writelines(lines[:1] + lines[2:])  # Keep the first line and all after the second
            file.truncate()  # Truncate the file to remove leftover data
            
            return second_line
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


N_SIMULATIONS_PER_SCENARIO = 25




simulations_config = {
    "BackdoorAdjustmentScenario": {
        "n_samples": [100, 200, 500, 1000, 2000, 5000],
        "d_c": [5, 10, 20, 50, 100],
        "d_a": [5, 10, 20, 50, 100],
        "noise_level_treatment": [0.1, 0.2, 0.5],
        "noise_level_target": [0.1, 0.2, 0.5],
        "alpha_corr_covariates": [0.3],
        "target_covariates_relationship_type": ["random"],
        "treatment_covariates_relationship_type": ["random"],
    },
    "InstrumentalVariableScenario": {
        "n_samples": [50, 100, 200, 500, 1000, 2000, 5000],
        "d_c": [5, 10, 20, 50, 100],
        "d_a": [5, 10, 20, 50, 100],
        "d_u": [5, 10, 20, 50],
        "noise_level_treatment": [0.1, 0.2, 0.5],
        "noise_level_target": [0.1, 0.2, 0.5],
        "noise_level_instrument": [0.1, 0.2, 0.5],
        "alpha_corr_covariates": [0.5],
        "target_covariates_relationship_type": ["random"],
        "treatment_covariates_relationship_type": ["random"],
        "instrument_covariates_relationship_type": ["random"],
    }
}

matches = {
    "BackdoorAdjustmentScenario": [
        lambda df: df.noise_level_treatment == df.noise_level_target,
        lambda df: df.target_covariates_relationship_type == df.treatment_covariates_relationship_type,
        lambda df: df.n_samples > 1.2 * (df.d_c + df.d_a),
        lambda df: (df.d_c >= .75 * df.d_a) & (df.d_a >= .75 * df.d_c),
    ],
    "InstrumentalVariableScenario": [
        lambda df: df.noise_level_treatment == df.noise_level_target,
        lambda df: df.noise_level_treatment == df.noise_level_instrument,
        lambda df: df.target_covariates_relationship_type == df.treatment_covariates_relationship_type,
        lambda df: df.target_covariates_relationship_type == df.instrument_covariates_relationship_type,
        lambda df: df.n_samples > 1.2 * (df.d_c + df.d_a),
        lambda df: (df.d_u >= df.d_c * .5) & (df.d_u <= df.d_c)
    ]
}


def create_combinations(simulations_config, matches, n_simulations_per_scenario):
    keys = list(simulations_config.keys())
    values = list(simulations_config.values())
    combinations = pd.DataFrame([
        dict(zip(keys, combination)) for combination in itertools.product(*values)
    ])
    for match in matches:
        combinations = combinations.loc[match]
    simulation_plan = pd.DataFrame()
    for i in range(n_simulations_per_scenario):
        combinations = (
            combinations
            .sort_values(
                ["alpha_corr_covariates", "noise_level_treatment", "n_samples"],
                ascending=[True, True, False, True]
            )
        )
        combinations['simulation_id'] = i + 1
        simulation_plan = pd.concat([simulation_plan, combinations])
    simulation_plan['completed'] = False
    simulation_plan['external_id'] = np.arange(len(simulation_plan)) + 1
    return (
        simulation_plan
        .reset_index(drop=True)
        .pipe(lambda df: pd.concat(
            [df[['external_id']], df.drop('external_id', axis=1)
        ], axis=1))
    )
    

if __name__ == "__main__":
    os.makedirs("simulations/plans", exist_ok=True)
    if os.path.exists("simulations/plans/backdoor_simulation_plan_c.txt"):
        input_overwrite = input("The backdoor_simulation_plan.txt file already exists. Do you want to overwrite it? (Yes/No) ")
    else:
        input_overwrite = "yes"
    if input_overwrite.lower() == "yes":
        backdoor_simulation_plan = create_combinations(
            simulations_config['BackdoorAdjustmentScenario'],
            matches['BackdoorAdjustmentScenario'],
            N_SIMULATIONS_PER_SCENARIO
        )
        backdoor_simulation_plan.to_csv("simulations/plans/backdoor_simulation_plan_c.txt", index=False)

    if os.path.exists("simulations/plans/instrumental_simulation_plan.txt"):
        input_overwrite = input("The instrumental_simulation_plan.txt file already exists. Do you want to overwrite it? (Yes/No) ")
    else:
        input_overwrite = "yes"
    if input_overwrite.lower() == "yes":
        instrumental_simulation_plan = create_combinations(
            simulations_config['InstrumentalVariableScenario'],
            matches['InstrumentalVariableScenario'],
            N_SIMULATIONS_PER_SCENARIO
        )
        instrumental_simulation_plan.to_csv("simulations/plans/instrumental_simulation_plan.txt", index=False)


