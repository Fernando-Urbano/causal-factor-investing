import pytest
import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.init.database_initialization import initialize_database
from scenarios.causal_scenarios import BackdoorAdjustmentScenario, SCHEMA_COLUMNS

@pytest.fixture(autouse=True)
def clean_test_db():
    yield
    conn = sqlite3.connect("database/test_causal_scenarios.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scenario_simulations")
    conn.commit()
    conn.close()

def test_initialize_test_db_if_not_exists():
    if not os.path.exists("database/test_causal_scenarios.db"):
        initialize_database("database/test_causal_scenarios.db")
        assert os.path.exists("database/test_causal_scenarios.db")

def test_save_data_to_test_db():
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

    scenario.save_in_test_db()
    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()
    scenario.save_summary()

    conn = sqlite3.connect("database/test_causal_scenarios.db")
    df = pd.read_sql_query("SELECT * FROM scenario_simulations WHERE scenario_name = 'BackdoorAdjustmentScenario'", conn)
    conn.close()
    assert len(df.index) == 1
    assert df.iloc[0]["scenario_name"] == "BackdoorAdjustmentScenario"
    assert df.iloc[0]["true_ate"] == scenario.true_ate
    assert df.iloc[0]["estimated_ate"] == scenario.get_summary()["estimated_ate"].iloc[0]


def test_save_data_with_missing_attributes():
    """
    Test that attempting to save a scenario with missing attributes raises an error.
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

    with pytest.raises(ValueError):
        scenario.save_summary()


def test_save_data_edge_case():
    """
    Test saving a scenario with edge case data such as no noise and high correlation.
    """
    scenario = BackdoorAdjustmentScenario(
        n_samples=25,
        d_c=5,
        d_a=3,
        noise_level_treatment=0.0,
        noise_level_target=0.0,
        alpha_corr_covariates=1.0,
        l_model='RF',
        m_model='RF',
        specification='Correct'
    )

    scenario.generate_data()
    scenario.build_model()
    scenario.fit_model()
    scenario.save_summary()

    conn = sqlite3.connect("database/test_causal_scenarios.db")
    df = pd.read_sql_query("SELECT * FROM scenario_simulations WHERE scenario_name = 'BackdoorAdjustmentScenario'", conn)
    conn.close()

    assert len(df.index) == 1
    assert df.iloc[0]["true_ate"] == scenario.true_ate


def test_database_schema_consistency():
    """
    Test that the database schema remains consistent when new scenarios are added.
    """
    conn = sqlite3.connect("database/test_causal_scenarios.db")
    cursor = conn.cursor()

    # Get schema info
    cursor.execute("PRAGMA table_info(scenario_simulations)")
    schema = cursor.fetchall()
    conn.close()

    schema_columns = [column[1] for column in schema]
    schema_columns.remove("id")
    assert set(schema_columns) == set(SCHEMA_COLUMNS)

if __name__ == "__main__":
    test_save_data_to_test_db()