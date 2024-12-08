# test_relationship_functions.py

import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.causal_scenarios import (
    NON_LINEAR_TRANSFORMATIONS,
    generate_non_linear_relationship,
    modify_treatment_effect_and_compute_ate,
    calc_true_ite,
    USE_SPECIAL_ITE_CALCULATION
)

def test_generate_non_linear_relationship():
    """
    Tests that generate_non_linear_relationship() returns an array of the correct shape
    and that the transformations applied are valid.
    """
    np.random.seed(42)
    X = np.random.normal(size=(50, 5))
    y = generate_non_linear_relationship(X, relationship_type="square")
    assert y.shape == (50,)
    assert not np.isnan(y).any()

    y_random, transforms = generate_non_linear_relationship(X, relationship_type="random", return_relationship_types=True)
    assert y_random.shape == (50,)
    assert isinstance(transforms, tuple)
    assert len(transforms) == 5
    for func in transforms:
        assert func in NON_LINEAR_TRANSFORMATIONS.values()


def test_modify_treatment_effect_and_compute_ate_constant_ite():
    """
    Tests that modify_treatment_effect_and_compute_ate() returns the correct shapes
    and handles constant ITE scenario correctly.
    """
    np.random.seed(42)
    treatment = np.random.binomial(1, p=0.5, size=100)
    treatment_effect, true_ate, relationship_types = modify_treatment_effect_and_compute_ate(
        treatment=treatment,
        relationship_type="linear",
        X_c=None,
        constant_ite=True,
        random_seed=42
    )
    assert treatment_effect.shape == (100,)
    assert np.allclose(treatment_effect[treatment == 0], 0.0)
    assert not np.isnan(true_ate)
    assert isinstance(relationship_types, (tuple, type(None)))


def test_modify_treatment_effect_and_compute_ate_non_constant_ite():
    """
    Tests that modify_treatment_effect_and_compute_ate() returns non-constant ITE when X_c is provided
    and that relationship_types is consistent.
    """
    np.random.seed(42)
    X_c = np.random.normal(size=(100, 5))
    treatment = np.random.binomial(1, p=0.5, size=100)
    treatment_effect, true_ate, relationship_types = modify_treatment_effect_and_compute_ate(
        treatment=treatment,
        relationship_type="square",
        X_c=X_c,
        constant_ite=False,
        random_seed=42
    )
    assert treatment_effect.shape == (100,)
    assert not np.isnan(true_ate)
    assert isinstance(relationship_types, tuple)
    # Check that not all treatment effects are identical
    assert not np.allclose(treatment_effect[treatment == 1], treatment_effect[treatment == 1][0])


def test_calc_true_ite():
    """
    Tests that calc_true_ite() correctly returns an array of true ITE values
    and a list of chosen relationships.
    """
    np.random.seed(42)
    X_c = np.random.normal(size=(100, 5))
    true_ite, relationships = calc_true_ite(X_c, relationship_type_interaction="random", n_subgroups=2, pct_cols_in_subgroup=0.5)
    assert true_ite.shape == (100,)
    assert len(relationships) == 2
    for rel in relationships:
        assert rel in NON_LINEAR_TRANSFORMATIONS.keys()


@pytest.mark.parametrize("use_special_ite", [True, False])
def test_modify_treatment_effect_and_compute_ate_with_special_ite_calculation(use_special_ite):
    """
    Tests that modify_treatment_effect_and_compute_ate() behaves correctly
    when USE_SPECIAL_ITE_CALCULATION is toggled.
    """
    original_setting = USE_SPECIAL_ITE_CALCULATION
    try:
        # Monkey patching the global variable for testing
        globals()['USE_SPECIAL_ITE_CALCULATION'] = use_special_ite
        np.random.seed(42)
        X_c = np.random.normal(size=(100, 5))
        treatment = np.random.binomial(1, p=0.5, size=100)
        treatment_effect, true_ate, relationship_types = modify_treatment_effect_and_compute_ate(
            treatment=treatment,
            relationship_type="cubic",
            X_c=X_c,
            constant_ite=False,
            random_seed=42
        )
        assert treatment_effect.shape == (100,)
        assert not np.isnan(true_ate)
        assert isinstance(relationship_types, tuple)
    finally:
        globals()['USE_SPECIAL_ITE_CALCULATION'] = original_setting


def test_generate_non_linear_relationship_sin_transform():
    """
    Tests generate_non_linear_relationship() with a fixed non-linear transformation ('sin')
    to ensure it produces expected output shapes and no NaNs.
    """
    np.random.seed(42)
    X = np.random.normal(size=(60, 4))
    y = generate_non_linear_relationship(X, relationship_type="sin")
    assert y.shape == (60,)
    assert not np.isnan(y).any()


def test_generate_non_linear_relationship_exponential_transform():
    """
    Tests generate_non_linear_relationship() with the 'exponential' transformation
    to ensure it works correctly with positive and negative values.
    """
    np.random.seed(42)
    X = np.random.uniform(-5, 5, size=(40, 3))
    y = generate_non_linear_relationship(X, relationship_type="exponential")
    assert y.shape == (40,)
    assert not np.isnan(y).any()
    # Check that the exponential transformation yields only positive values
    assert np.all(y > 0)


def test_modify_treatment_effect_and_compute_ate_no_X_c_non_constant():
    """
    Tests modify_treatment_effect_and_compute_ate() when no X_c is provided and constant_ite=False.
    This scenario should still produce valid outputs, falling back to a random constant ITE scenario.
    """
    np.random.seed(42)
    treatment = np.random.binomial(1, p=0.5, size=100)
    treatment_effect, true_ate, relationship_types = modify_treatment_effect_and_compute_ate(
        treatment=treatment,
        relationship_type="cubic",
        X_c=None,
        constant_ite=False,
        random_seed=42
    )
    assert treatment_effect.shape == (100,)
    assert not np.isnan(true_ate)
    # relationship_types should be None since no X_c is provided
    assert relationship_types is None


def test_calc_true_ite_full_columns():
    """
    Tests calc_true_ite() with pct_cols_in_subgroup=1 to ensure it uses all columns in each subgroup.
    """
    np.random.seed(42)
    X_c = np.random.normal(size=(80, 5))
    true_ite, relationships = calc_true_ite(X_c, relationship_type_interaction="linear", n_subgroups=3, pct_cols_in_subgroup=1)
    assert true_ite.shape == (80,)
    assert len(relationships) == 3
    # When using 'linear', no error should occur and relationships should all be 'linear'
    assert all(rel == "linear" for rel in relationships)


def test_calc_true_ite_non_random_fixed_relationship():
    """
    Tests calc_true_ite() with a fixed non-random relationship_type_interaction (e.g. 'cos')
    and checks that the returned relationships match the chosen type.
    """
    np.random.seed(42)
    X_c = np.random.normal(size=(100, 5))
    true_ite, relationships = calc_true_ite(X_c, relationship_type_interaction="cos", n_subgroups=2, pct_cols_in_subgroup=0.5)
    assert true_ite.shape == (100,)
    assert len(relationships) == 2
    for rel in relationships:
        assert rel == "cos"