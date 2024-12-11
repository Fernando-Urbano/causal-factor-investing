# test_model_fitting.py

import pytest
import numpy as np
import os
import sys

# Adjust the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.causal_scenarios import (
    BackdoorAdjustmentScenario,
    InstrumentalVariableScenario,
    DATA_GENERATION_SPECIFICATION
)

def test_generate_data_multiple_times():
    scenario = BackdoorAdjustmentScenario(
        n_samples=1000,
        d_c=1,
        d_a=3,
        noise_level_treatment=0.0,
        noise_level_target=0.0,
        alpha_corr_covariates=0.0,
        target_covariates_relationship_type = "linear",
        treatment_covariates_relationship_type = "linear",
        l_model='LASSO',
        m_model='DNN',
        specification='Correct',
        seed_data=42
    )
    for _ in range(10):
        scenario.generate_data()
        scenario.build_model()


def test_generate_data_for_different_specifications_in_backdoor():
    scenario = BackdoorAdjustmentScenario(
        n_samples=1000,
        d_c=3,
        d_a=3,
        noise_level_treatment=0.0,
        noise_level_target=0.0,
        alpha_corr_covariates=0.0,
        target_covariates_relationship_type = "linear",
        treatment_covariates_relationship_type = "linear",
        l_model='LASSO',
        m_model='DNN',
        specification='Correct',
        seed_data=42
    )
    for specification in DATA_GENERATION_SPECIFICATION['BackdoorAdjustmentScenario']:
        scenario.set_specification(specification)
        scenario.generate_data()
        scenario.build_model()


def test_generate_data_for_different_specifications_in_iv():
    scenario = InstrumentalVariableScenario(
        n_samples=1000,
        d_c=6,
        d_a=3,
        d_u=3,
        noise_level_instrument=0.0,
        noise_level_target=0.0,
        noise_level_treatment=0.0,
        alpha_corr_covariates=0.0,
        l_model="DNN",
        m_model="DNN",
        r_model="DNN",
        seed_data=42
    )
    for specification in DATA_GENERATION_SPECIFICATION['InstrumentalVariableScenario']:
        scenario.set_specification(specification)
        scenario.generate_data()
        scenario.build_model()

def test_generate_data_for_extra_unobserved():
    scenario = InstrumentalVariableScenario(
        n_samples=1000,
        d_c=6,
        d_a=3,
        d_u=3,
        noise_level_instrument=0.0,
        noise_level_target=0.0,
        noise_level_treatment=0.0,
        alpha_corr_covariates=0.0,
        l_model="DNN",
        m_model="DNN",
        r_model="DNN",
        seed_data=42,
        specification='Extra Unobserved Confounders'
    )
    for i in range(10):
        scenario.set_specification('Extra Unobserved Confounders')
        scenario.set_pct_extra_unobserved(max(i / 10, 0.1))
        scenario.generate_data()
        scenario.build_model()
        scenario.set_specification('Correct')
        scenario.get_pct_extra_unobserved() == max(i / 10, 0.1)

def test_generate_data_for_different_specifications_in_iv_with_one_feature():
    scenario = InstrumentalVariableScenario(
        n_samples=1000,
        d_c=1,
        d_a=1,
        d_u=0,
        noise_level_instrument=0.0,
        noise_level_target=0.0,
        noise_level_treatment=0.0,
        alpha_corr_covariates=0.0,
        l_model="DNN",
        m_model="DNN",
        r_model="DNN",
        seed_data=42
    )
    for specification in DATA_GENERATION_SPECIFICATION['InstrumentalVariableScenario']:
        scenario.set_specification(specification)
        scenario.generate_data()
        scenario.build_model()


if __name__ == "__main__":
    test_generate_data_multiple_times()