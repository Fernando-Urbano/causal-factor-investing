# test_model_fitting.py

import pytest
import numpy as np
import os
import sys

# Adjust the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.causal_scenarios import (
    BackdoorAdjustmentScenario,
    InstrumentalVariableScenario
)

def test_backdoor_adjustment_correct_spec():
    """
    Tests fitting the BackdoorAdjustmentScenario with the 'Correct' specification.
    Should successfully build and fit the model.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=50,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.5,
        l_model='RF',
        m_model='RF',
        specification='Correct'
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()

    assert scenario.dml is not None
    assert isinstance(scenario.get_summary()['estimated_ate'].iloc[0], (float, int))
    assert isinstance(scenario.true_ate, (float, int))


def test_backdoor_adjustment_correct_spec_no_noise():
    scenario = BackdoorAdjustmentScenario(
        n_samples=1000,
        d_c=1,
        d_a=3,
        noise_level_treatment=0.0,
        noise_level_target=0.0,
        alpha_corr_covariates=0.0,
        target_covariates_relationship_type = "linear",
        treatment_covariates_relationship_type = "linear",
        l_model='DNN',
        m_model='DNN',
        specification='Correct',
        seed_data=42
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()

    assert scenario.dml is not None
    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert abs(estimated_ate - scenario.true_ate) < 0.5


def test_backdoor_adjustment_with_regenerated_data():
    """
    Tests fitting the BackdoorAdjustmentScenario with the 'Correct' specification.
    Should successfully build and fit the model.
    """
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
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()

    first_true_ate = scenario.true_ate

    first_estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    
    scenario.build_model()
    scenario.fit_model()

    second_estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert abs(first_estimated_ate - second_estimated_ate) < .01

    scenario.generate_data()
    scenario.build_model()

    assert first_true_ate != scenario.true_ate


def test_fit_two_models_with_same_data_backdoor_scenario():
    scenario = BackdoorAdjustmentScenario(
        n_samples=200,
        d_c=3,
        d_a=5,
        noise_level_treatment=1,
        noise_level_target=1,
        alpha_corr_covariates=1,
        l_model='LASSO',
        m_model='RF',
        specification='Unobserved Confounders',
        seed_data=123,
        pct_unobserved=.7
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()
    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    scenario.set_l_model('DNN')
    scenario.set_m_model('DNN')
    scenario.build_model()
    scenario.fit_model()
    new_estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert estimated_ate != new_estimated_ate


def test_fit_two_models_with_same_data_iv_scenario():
    scenario = InstrumentalVariableScenario(
        n_samples=1000,
        d_c=6,
        d_a=3,
        d_u=3,
        noise_level_instrument=1,
        noise_level_target=1,
        noise_level_treatment=1,
        alpha_corr_covariates=1,
        l_model="RF",
        m_model="RF",
        r_model="RF",
        seed_data=42,
        specification='Extra Unobserved Confounders'
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()
    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    scenario.set_l_model('DNN')
    scenario.set_m_model('DNN')
    scenario.set_r_model('DNN')
    scenario.build_model()
    scenario.fit_model()
    new_estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert estimated_ate != new_estimated_ate

def test_backdoor_adjustment_inclusion_non_causal_cofounders():
    """
    Tests fitting the BackdoorAdjustmentScenario with the 'Inclusion of Non-Causal Cofounders' specification.
    Ensures the model can be built, fit, and the estimated ATE is retrieved.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=200,
        d_c=3,
        d_a=5,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.3,
        l_model='LASSO',
        m_model='RF',
        specification='Inclusion of Non-Causal Cofounders',
        seed_data=123
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()

    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert scenario.dml is not None
    assert isinstance(estimated_ate, (float, int))


def test_backdoor_adjustment_unobserved_confounders():
    """
    Tests fitting the BackdoorAdjustmentScenario with 'Unobserved Confounders' specification.
    Checks that the model runs and that the resulting estimated ATE is a numeric value.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=300,
        d_c=5,
        d_a=2,
        noise_level_treatment=0.05,
        noise_level_target=0.05,
        alpha_corr_covariates=0.2,
        l_model='EN',
        m_model='DNN',
        specification='Unobserved Confounders',
        constant_ite=False,
        seed_data=50
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()

    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert isinstance(estimated_ate, float)


def test_instrumental_variable_correct_binary_instrument():
    """
    Tests fitting the InstrumentalVariableScenario with a binary instrument and the 'Correct' specification.
    Ensures that the scenario can build and fit a DoubleMLIIVM model and produces a valid estimated ATE.
    """
    scenario = InstrumentalVariableScenario(
        n_samples=200,
        d_c=3,
        d_a=2,
        d_u=1,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        noise_level_instrument=0.05,
        alpha_corr_covariates=0.3,
        l_model='RF',
        m_model='RF',
        r_model='EN',
        binary_instrument=True,
        specification='Correct',
        seed_data=10
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.dml.fit()
    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert scenario.dml is not None
    assert isinstance(estimated_ate, float)


def test_instrumental_variable_instrument_as_cofounder():
    """
    Tests fitting the InstrumentalVariableScenario with 'Instrument treated as Cofounder' specification.
    Validates that a DoubleMLPLR model is created and estimated, and that a numeric ATE is obtained.
    """
    scenario = InstrumentalVariableScenario(
        n_samples=150,
        d_c=4,
        d_a=4,
        d_u=1,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        noise_level_instrument=0.1,
        alpha_corr_covariates=0.4,
        l_model='DNN',
        m_model='EN',
        r_model='LASSO',
        binary_instrument=True,
        specification='Instrument treated as Cofounder',
        seed_data=99
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.dml.fit()
    estimated_ate = scenario.get_summary()['estimated_ate'].iloc[0]
    assert isinstance(estimated_ate, float)


def test_backdoor_adjustment_bootstrap():
    """
    Tests fitting a BackdoorAdjustmentScenario and then performing a bootstrap to obtain confidence intervals.
    Checks that the bootstrap method runs and updates the summary with confidence intervals.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=100,
        d_c=5,
        d_a=2,
        noise_level_treatment=0.1,
        noise_level_target=0.1,
        alpha_corr_covariates=0.1,
        l_model='RF',
        m_model='LASSO',
        specification='Correct',
        seed_data=1234
    )
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()
    scenario.dml.bootstrap(method='normal', n_rep_boot=100)

    df_summary = scenario.get_summary()
    assert 'ci_2_5_pct' in df_summary.columns
    assert 'ci_97_5_pct' in df_summary.columns
    assert not np.isnan(df_summary['ci_2_5_pct'].iloc[0])
    assert not np.isnan(df_summary['ci_97_5_pct'].iloc[0])


if __name__ == "__main__":
    test_backdoor_adjustment_correct_spec_no_noise()
    test_backdoor_adjustment_with_regenerated_data()
    test_backdoor_adjustment_correct_spec()
    test_backdoor_adjustment_inclusion_non_causal_cofounders()
    test_backdoor_adjustment_unobserved_confounders()
    test_instrumental_variable_correct_binary_instrument()
    test_instrumental_variable_instrument_as_cofounder()
    test_backdoor_adjustment_bootstrap()