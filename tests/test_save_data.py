import pytest
import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.init.database_initialization import initialize_database
from scenarios.causal_scenarios import BackdoorAdjustmentScenario

@pytest.fixture(autouse=True)
def clean_test_db():
    yield
    # conn = sqlite3.connect("database/test_causal_scenarios.db")
    # cursor = conn.cursor()
    # cursor.execute("DELETE FROM scenario_simulations")
    # conn.commit()
    # conn.close()

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


if __name__ == "__main__":
    test_save_data_to_test_db()