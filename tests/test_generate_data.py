# test_simulation_creation.py

import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.causal_scenarios import (
    CausalScenario,
    BackdoorAdjustmentScenario,
    InstrumentalVariableScenario
)

def test_causal_scenario_generate_x():
    """
    Tests that _generate_X() correctly generates X_c and X_a with expected shapes.
    """
    scenario = CausalScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.5,
        l_model='DNN',
        m_model='RF',
    )
    scenario._generate_X()
    assert scenario.X_c.shape == (100, 5)
    assert scenario.X_a.shape == (100, 3)
    assert scenario.X_c is not None
    assert scenario.X_a is not None


def test_causal_scenario_generate_data():
    """
    Tests that generate_data() correctly produces treatment and target arrays.
    """
    scenario = CausalScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.5,
        l_model='DNN',
        m_model='RF'
    )
    scenario.generate_data()
    assert scenario.treatment.shape == (100,)
    assert scenario.target.shape == (100,)
    assert scenario.true_ate is not None
    assert not np.isnan(scenario.true_ate)


def test_backdoor_scenario_generate_data_correct_spec():
    """
    Tests that the BackdoorAdjustmentScenario in the correct specification
    produces valid treatment and target arrays and that no extra variables are introduced.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.5,
        specification='Correct',
        l_model='DNN',
        m_model='RF',
    )
    scenario.generate_data()
    assert scenario.treatment.shape == (100,)
    assert scenario.target.shape == (100,)
    assert scenario.true_ate is not None
    assert scenario.X_c.shape == (100, 5)
    assert scenario.X_a.shape == (100, 3)


def test_backdoor_scenario_generate_data_unobserved_conf():
    """
    Tests that the BackdoorAdjustmentScenario with unobserved confounders
    correctly adds extra features to X and still produces valid treatment and target.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.5,
        specification='Unobserved Confounders',
        l_model='DNN',
        m_model='RF',
    )
    scenario.generate_data()
    scenario.build_model()
    assert scenario.treatment.shape == (100,)
    assert scenario.target.shape == (100,)
    # After build_model(), check if X in the model is expanded
    n_unobserved = int(np.ceil(5 * scenario.pct_unobserved))
    expected_dim = 5 + n_unobserved + 3
    # The DoubleMLData object is stored in scenario.dml, we can inspect scenario.dml.data
    assert scenario.dml._dml_data.x.shape[1] == expected_dim


def test_instrumental_variable_scenario_generate_data_correct_spec():
    """
    Tests the InstrumentalVariableScenario in the correct specification,
    ensuring that U is removed and Z is appropriately generated.
    """
    scenario = InstrumentalVariableScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        d_u=2,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        noise_level_instrument=0.1,
        alpha_corr_covariates=0.5,
        specification='Correct',
        l_model='DNN',
        m_model='RF',
        r_model='DNN'
    )
    scenario.generate_data()
    assert scenario.treatment.shape == (100,)
    assert scenario.target.shape == (100,)
    assert scenario.Z is not None
    assert scenario.U.shape == (100, 2)
    assert scenario.X_c_ex_U.shape == (100, 3)  # d_c - d_u = 5 - 2 = 3


def test_instrumental_variable_scenario_generate_data_extra_unobserved():
    """
    Tests the InstrumentalVariableScenario with Extra Unobserved Confounders,
    ensuring that additional random features are correctly introduced.
    """
    scenario = InstrumentalVariableScenario(
        n_samples=100,
        d_c=5,
        d_a=3,
        d_u=1,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        noise_level_instrument=0.1,
        alpha_corr_covariates=0.5,
        specification='Extra Unobserved Confounders',
        pct_extra_unobserved=0.4,
        l_model='DNN',
        m_model='RF',
        r_model='DNN'
    )
    scenario.generate_data()
    scenario.build_model()
    n_unobserved = int(np.ceil(5 * 0.4))
    # d_c_ex_U = d_c - d_u = 5 - 1 = 4
    expected_dim = 4 + n_unobserved + 3
    # Check dimension of X in scenario.dml data
    assert scenario.dml._dml_data.x.shape[1] == expected_dim

if __name__ == "__main__":
    pytest.main()