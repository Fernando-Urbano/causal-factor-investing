import pytest
import numpy as np
import os
import pandas as pd
import sys
import portalocker

# Adjust the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.causal_scenarios import (
    BackdoorAdjustmentScenario,
    InstrumentalVariableScenario,
    DATA_GENERATION_SPECIFICATION,
    get_all_pipeline_names
)

PLAN_INFO_NAMES = [
    'external_id',
    'n_samples',
    'd_c',
    'd_a',
    'noise_level_treatment',
    'noise_level_target',
    'alpha_corr_covariates',
    'target_covariates_relationship_type',
    'treatment_covariates_relationship_type',
    'simulation_id',
    'completed'
]

from simulations.simulations_plan import read_and_delete_second_line

def keep_simulations():
    with open("simulations/keep_simulations.txt", "r") as f:
        keep_simulations = f.read().splitlines()
    return keep_simulations[0] == "True"

if __name__ == "__main__":
    seed = np.random.randint(1, 100001)
    while True:
        if not keep_simulations():
            break
        simulation_info = read_and_delete_second_line("simulations/plans/backdoor_simulation_plan.txt")
        simulation_info = dict(zip(PLAN_INFO_NAMES, simulation_info.split(",")))
        external_id = int(simulation_info['external_id'])
        print(f"Started BackdoorAdjustmentScenario S{external_id:.0f}")
        scenario = BackdoorAdjustmentScenario(
            n_samples=int(simulation_info['n_samples']),
            d_c=int(simulation_info['d_c']),
            d_a=int(simulation_info['d_a']),
            noise_level_treatment=float(simulation_info['noise_level_treatment']),
            noise_level_target=float(simulation_info['noise_level_target']),
            alpha_corr_covariates=float(simulation_info['alpha_corr_covariates']),
            target_covariates_relationship_type=simulation_info['target_covariates_relationship_type'],
            treatment_covariates_relationship_type=simulation_info['treatment_covariates_relationship_type'],
            seed_data=seed
        )
        scenario.generate_data()
        for specification in DATA_GENERATION_SPECIFICATION['BackdoorAdjustmentScenario']:
            scenario.set_specification(specification)
            for model in get_all_pipeline_names("RF"):
                time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{time_stamp} BackdoorAdjustmentScenario S{external_id:.0f}: "
                    + f"l_model='{model}', m_model='{model}', specification='{specification}'"
                )
                scenario.set_l_model(model)
                scenario.set_m_model(model)
                scenario.build_model()
                scenario.fit_model()
                scenario.save_summary()


        



