import numpy as np
import sqlite3
import pandas as pd
from typing import Union
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLPLIV
from doubleml.double_ml import DoubleML
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, LogisticRegressionCV, ElasticNetCV
import datetime
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import random


SEED_MODEL = 42
USE_SPECIAL_ITE_CALCULATION = False


DATA_GENERATION_SPECIFICATION = [
    'Correct',
    'Inclusion of Non-Causal Cofounders',
    'Unobserved Confounders',
    'Extra Unobserved Confounders',
    'Instrument treated as Cofounder',
    'Instrument treated as Cofounder and Inclusion of Non-Causal Cofounders'
]


SCHEMA_COLUMNS = [
    "scenario_name", "seed_data", "n_samples", "d_c", "d_a", "timestamp_data_generation",
    "noise_level_treatment", "noise_level_target", "alpha_corr_covariates", "true_ate", "avg_corr_covariates",
    "l_model", "m_model", "cv_loss_regression", "cv_loss_classification",
    "specification", "target_covariates_relationship_type",
    "treatment_covariates_relationship_type", "estimated_ate", "std_error",
    "ci_2_5_pct", "ci_97_5_pct"
]

NON_LINEAR_TRANSFORMATIONS = {
    "linear": lambda t: t,
    "sin": lambda t: np.sin(t),
    "cos": lambda t: np.cos(t),
    "square": lambda t: t**2,
    "cubic": lambda t: t**3,
    "step": lambda t: np.where(t > 0, 1, -1),
    "piecewise_linear": lambda t: np.piecewise(
        t, [t < 0, (t >= 0) & (t < 1), t >= 1],
        [lambda x: x * 2, lambda x: x, lambda x: x * 0.5]
    ),
    "exponential": lambda t: np.exp(t / 5),
    "logarithmic": lambda t: np.log1p(np.abs(t)),
}


PARAM_GRID = {
    "RF": {
        'model__n_estimators': [50, 100, 200, 500],
        'model__max_depth': [2, 3, 5, 10]
    },
    "DNN": {
        'model__hidden_layer_sizes': [
            (64, 64, 64, 32, 16,),
            (32, 32, 32, 32, 32,),
            (128, 64, 32, 16,),
            (64, 64, 64, 64),
            (128, 64, 32,),
            (32, 32, 32,),
            (32, 32,),
            (32,)
        ],
        'model__alpha': [.001, .01, .1, .2, .5, 1],  # Ridge regularization
    }
}


NORMAL_PIPELINES = {
    "classification": {
        "LASSO": [
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegressionCV(
                cv=5,
                penalty='l1',
                solver='saga',
                random_state=SEED_MODEL,
                max_iter=10000
            ))
        ],
        "EN": [
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegressionCV(
                cv=5,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0],
                random_state=SEED_MODEL,
                max_iter=10000
            ))
        ],
        "RF": [('model', RandomForestClassifier(random_state=SEED_MODEL))],
        "DNN": [
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(max_iter=1000, random_state=SEED_MODEL, activation='relu', learning_rate='adaptive'))
        ]
    },
    "regression": {
        "LASSO": [
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LassoCV(cv=5, random_state=SEED_MODEL))
        ],
        "EN": [
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', ElasticNetCV(
                cv=5,
                l1_ratio=[0, 0.1, 0.5, 0.7, 0.9, 1.0],
                random_state=SEED_MODEL
            ))
        ],
        "RF": [
            ('model', RandomForestRegressor(random_state=SEED_MODEL))
        ],
        "DNN": [
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(max_iter=1000, random_state=SEED_MODEL, activation='relu', learning_rate='adaptive'))
        ]
    }
}


def add_grid_search(pipeline, model, scoring):
    if model in PARAM_GRID.keys():
        return GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRID[model],
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
    else:
        return pipeline



CV_DEFAULT_LOSS_REGRESSION = 'neg_mean_squared_error'
CV_DEFAULT_LOSS_CLASSIFICATION = 'balanced_accuracy'


def covariance_to_correlation(cov_matrix: np.matrix) -> np.matrix:
    """
    Converts a covariance matrix to a correlation matrix.

    Parameters:
    - cov_matrix (np.ndarray): Covariance matrix (n_features x n_features).

    Returns:
    - corr_matrix (np.ndarray): Correlation matrix (n_features x n_features).
    """
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(corr_matrix, 1)
    return corr_matrix


def calc_avg_corr(cov_matrix):
    corr_matrix = covariance_to_correlation(cov_matrix)
    corr_matrix_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_no_diag, np.nan)
    return np.nanmean(corr_matrix_no_diag)


def generate_non_linear_relationship(
        X: np.matrix,
        relationship_type: str = "random",
        random_seed: int = None,
        instrument_is_the_last_covariate: bool = False,
        return_relationship_types: bool = False
    ):
    """
    Creates a non-linear relationship between covariates X and a target.

    Parameters:
    - X (np.ndarray): Covariates (n_samples x n_features).
    - relationship_type (str): Type of non-linear relationship. Options:
        - "sin", "cos", "square", "cubic", "step",
          "piecewise_linear", "exponential", "logarithmic", "random".
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - y (np.ndarray): Target values with non-linear relationship.
    """
    random_seed = SEED_MODEL if random_seed is None else random_seed
    np.random.seed(random_seed)

    n_features = X.shape[1]

    if relationship_type == "random":
        selected_transforms = np.random.choice(list(NON_LINEAR_TRANSFORMATIONS.values()), size=n_features)
    elif relationship_type in NON_LINEAR_TRANSFORMATIONS:
        selected_transforms = [NON_LINEAR_TRANSFORMATIONS[relationship_type]] * n_features
    else:
        raise ValueError(
            f"Invalid relationship_type '{relationship_type}'. Must be one of {list(NON_LINEAR_TRANSFORMATIONS.keys())}."
        )

    transformed_features = np.column_stack(
        [selected_transforms[i](X[:, i]) for i in range(n_features)]
    )
    
    if not instrument_is_the_last_covariate:
        weights = np.random.uniform(-1, 1, size=n_features)
    else:
        weights = np.random.uniform(-1, 1, size=n_features - 1)
        weights = np.append(weights, 1)
    y = transformed_features @ weights

    if return_relationship_types:
        return y, tuple(selected_transforms)
    return y


def modify_treatment_effect_and_compute_ate(
        treatment: np.array,
        relationship_type: str = "random",
        X_c: Union[np.ndarray, np.matrix] = None,
        constant_ite : bool = False,
        random_seed: int = None
    ):
    """
    Modifies the treatment effect with a variety of non-linear relationships and computes the true ATE.

    Parameters:
    - treatment (np.ndarray): Array of treatment assignments (0 or 1).
    - relationship_type_treatment (str): Type of non-linear transformation for the treatment effect.
    - relationship_type_target (str): Type of non-linear transformation for interaction with covariates.
    - X_c (np.ndarray): Covariate matrix (n_samples x n_features).
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - treatment_effect (np.ndarray): Treatment effect for each individual (tau(X_i) * T_i).
    - true_ate (float): True average treatment effect across all individuals.
    - relationship_type_treatment (str): The chosen relationship type for treatment effect.
    - relationship_type_target (str): The chosen relationship type for interaction effect.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if relationship_type not in NON_LINEAR_TRANSFORMATIONS:
        raise ValueError(
            f"Invalid relationship_type_target '{relationship_type}'. "
            f"Must be one of {list(NON_LINEAR_TRANSFORMATIONS.keys())}."
        )

    if X_c is None or constant_ite:
        true_ate = np.random.normal(loc=0, scale=1) + 2
        true_ite = true_ate * np.ones(treatment.shape[0])
    else:
        if USE_SPECIAL_ITE_CALCULATION:
            true_ite, relationship_types = calc_true_ite(X_c, relationship_type, 4)
        generate_non_linear_relationship(X_c, relationship_type)
        true_ate = np.mean(true_ite)

    treatment_effect = treatment * true_ite

    return treatment_effect, true_ate, tuple(relationship_types)


def calc_true_ite(
    X_c: Union[np.ndarray, np.matrix],
    relationship_type_interaction: str,
    n_subgroups: int = 4,
    pct_cols_in_subgroup: float = 0.6
) -> np.ndarray:
    """
    Calculates the Individual Treatment Effect (ITE) based on non-linear interactions with covariates.

    Parameters:
    - X_c (Union[np.ndarray, np.matrix]): Covariate matrix (n_samples x n_features).
    - relationship_type_interaction (str): Type of non-linear transformation for interactions.
    - n_subgroups (int): Number of subgroups to create interactions.
    - pct_cols_in_subgroup (float): Percentage of covariates to include in each subgroup.

    Returns:
    - np.ndarray: True ITE for each individual.
    """
    if isinstance(X_c, np.matrix):
        X_c = np.array(X_c)
    
    n_covariates = X_c.shape[1]
    if pct_cols_in_subgroup == 1:
        n_covariates_subgroup = n_covariates
    else:
        n_covariates_subgroup = int(np.ceil(n_covariates * pct_cols_in_subgroup))
    true_ite = np.zeros(X_c.shape[0])

    relationship_types = []

    for _ in range(n_subgroups):
        covariates_cols_subgroup = random.sample(range(n_covariates), n_covariates_subgroup)
        
        if relationship_type_interaction == "random":
            chosen_relationship = random.choice(list(NON_LINEAR_TRANSFORMATIONS.keys()))
        else:
            chosen_relationship = relationship_type_interaction
        relationship_types.append(chosen_relationship)
        
        interaction_non_linear_function = NON_LINEAR_TRANSFORMATIONS[chosen_relationship]
        
        summed_covariates = X_c[:, covariates_cols_subgroup].sum(axis=1)
        interaction_effect = interaction_non_linear_function(summed_covariates)
        
        true_ite += interaction_effect.squeeze()

    return true_ite, relationship_types



class CausalScenario:
    db_path = "database/causal_scenarios.db"

    def __init__(
            self,
            n_samples: int, d_c: int, d_a: int,
            noise_level_treatment: float,
            noise_level_target: float,
            alpha_corr_covariates: float,
            l_model: str = None,
            m_model: str = None,
            cv_loss_regression: str = CV_DEFAULT_LOSS_REGRESSION,
            cv_loss_classification: str = CV_DEFAULT_LOSS_CLASSIFICATION,
            specification: str = 'correct',
            constant_ite: bool = False,
            seed_data: int = 42,
        ):
        self.n_samples = n_samples
        self.d_c = d_c
        self.d_a = d_a
        self.timestamp_data_generation = None
        self.noise_level_treatment = noise_level_treatment
        self.noise_level_target = noise_level_target
        self.alpha_corr_covariates = alpha_corr_covariates
        self.true_ate = None
        self.X_c = None
        self.X_a = None
        self.target = None
        self.treatment = None
        self.dml = None
        self.avg_corr_covariates = None
        self.l_model = l_model
        self.m_model = m_model
        self.cv_loss_regression = cv_loss_regression
        self.cv_loss_classification = cv_loss_classification
        self.specification = specification
        self.target_covariates_relationship_type = "random"
        self.treatment_covariates_relationship_type = "random"
        self.summary = None
        self.seed_data = seed_data
        self.constant_ite = constant_ite

        np.random.seed(self.seed_data)
        random.seed(self.seed_data)

    @classmethod
    def save_in_test_db(cls):
        cls.db_path = "database/test_causal_scenarios.db"
        return True

    @classmethod
    def save_in_main_db(cls):
        cls.db_path = "database/causal_scenarios.db"
        return True
    
    def _generate_X(self):
        self._set_generate_data_timestamp()
        d = self.d_c + self.d_a

        cov_matrix = make_sparse_spd_matrix(d, alpha=self.alpha_corr_covariates)
        self.avg_corr_covariates = calc_avg_corr(cov_matrix)

        X = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_matrix, size=self.n_samples)
        self.X_c = X[:, :self.d_c]
        self.X_a = X[:, self.d_c:]
        assert X.shape[1] == self.X_c.shape[1] + self.X_a.shape[1], "Invalid sizes for X_a and X_c"

    def generate_data(self):
        self._generate_X()

        noise_treatment = np.random.normal(scale=self.noise_level_treatment, size=self.n_samples)
        noise_target = np.random.normal(scale=self.noise_level_target, size=self.n_samples)

        logits = generate_non_linear_relationship(self.X_c, relationship_type="random") + noise_treatment
        prob_treatment = 1 / (1 + np.exp(-logits))
        self.treatment = np.random.binomial(1, p=prob_treatment)

        treatment_effect, self.true_ate, self.treatment_covariates_relationship_type = (
            modify_treatment_effect_and_compute_ate(
                treatment=self.treatment,
                relationship_type=self.treatment_covariates_relationship_type,
                X_c=self.X_c,
                constant_ite=False,
                random_seed=self.seed_data
            )
        )

        y0 = generate_non_linear_relationship(
            self.X_c, relationship_type=self.target_covariates_relationship_type
        )
        
        self.target = y0 + treatment_effect + noise_target


    def set_specification(self, new_specification):
        if new_specification not in DATA_GENERATION_SPECIFICATION:
            data_generation_spec = [f"({n}) {s}" for n, s in enumerate(DATA_GENERATION_SPECIFICATION)]
            raise ValueError(
                f"Invalid specification '{new_specification}'. "
                + f"Specification must be one of the following: {", ".join(data_generation_spec)}"
            )
        self.specification = new_specification
        return True
    
    def set_treatment_target_relationship_type(self, new_relationship):
        self.treatment_target_relationship_type = new_relationship
        return True

    def build_model(self):
        pass

    def _set_generate_data_timestamp(self) -> None:
        self.timestamp_data_generation = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def _build_summary(self) -> bool:
        if self.dml is None:
            raise ValueError("dml_plr is None")

        summary = self.dml.summary
        summary.reset_index(drop=True, inplace=True)
        summary.rename({
            "coef": "estimated_ate",
            "std err": "std_error",
            "2.5 %": "ci_2_5_pct",
            "97.5 %": "ci_97_5_pct"
        }, axis=1, inplace=True)

        model_information = {}
        for column in SCHEMA_COLUMNS:
            if column == "scenario_name":
                model_information[column] = self.__class__.__name__
            elif hasattr(self, column):
                model_information[column] = getattr(self, column)
            else:
                model_information[column] = pd.NA
                
        self.summary = pd.concat([summary, pd.DataFrame([model_information])], axis=1)

        return True


    def get_summary(self) -> pd.DataFrame:
        self._build_summary()
        return self.summary
    
    def save_summary(self) -> bool:
        """
        Saves the summary of the current scenario to an SQLite database.
        """
        self._build_summary()
        self.summary["scenario_name"] = self.__class__.__name__
        with sqlite3.connect(self.__class__.db_path) as conn:
            self.summary.to_sql(
                name='database/scenario_simulations',
                con=conn,
                if_exists='append',
                index=False
            )
        return True


class BackdoorAdjustmentScenario(CausalScenario):
    def __init__(
            self,
            n_samples: int,
            d_c: int,
            d_a: int,
            noise_level_treatment: float,
            noise_level_target: float,
            alpha_corr_covariates: float,
            l_model: str = None,
            m_model: str = None,
            cv_loss_regression: str = CV_DEFAULT_LOSS_REGRESSION,
            cv_loss_classification: str = CV_DEFAULT_LOSS_CLASSIFICATION,
            specification: str = 'Correct',
            pct_unobserved: float = 0.25,
            constant_ite: bool = False,
            seed_data: int = 42,
    ):
        super().__init__(
            n_samples,
            d_c,
            d_a,
            noise_level_treatment,
            noise_level_target,
            alpha_corr_covariates,
            l_model,
            m_model,
            cv_loss_regression,
            cv_loss_classification,
            specification,
            constant_ite,
            seed_data
        )
        self.pct_unobserved = pct_unobserved if specification == 'Unobserved Confounders' else None

    def build_model(self) -> bool:
        if self.target is None:
            raise ValueError('Data has not been generated')
        
        if self.specification == 'Correct':
            X = self.X_c
        elif self.specification == 'Inclusion of Non-Causal Cofounders':
            X = np.hstack((self.X_c, self.X_a))
        elif self.specification == 'Unobserved Confounders':
            n_unobserved = np.ceil(self.d_c * self.pct_unobserved)
            X_c_partial = np.hstack((self.X_c, np.random.normal(size=(self.n_samples, n_unobserved))))
            X = np.hstack((X_c_partial, self.X_a))
        else:
            raise ValueError(f"Unhandled specification: {self.specification}")

        dml_data = DoubleMLData.from_arrays(x=X, y=self.target, t=self.treatment)

        pipeline_l = Pipeline(NORMAL_PIPELINES['regression'][self.l_model])
        pipeline_m = Pipeline(NORMAL_PIPELINES['classification'][self.m_model])

        ml_l = add_grid_search(pipeline=pipeline_l, model=self.l_model, scoring=self.cv_loss_regression)
        ml_m = add_grid_search(pipeline=pipeline_m, model=self.m_model, scoring=self.cv_loss_classification)

        self.dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)

        return True

    def fit_model(self) -> bool:
        self.dml.fit()
        return True
        

class FrontdoorAdjustmentScenario(CausalScenario):
    def __init__(self, n_samples, d_c, d_a, noise_level, alpha_corr_covariates):
        super().__init__(n_samples, d_c, d_a, noise_level, alpha_corr_covariates)

    def generate_data(self):
        pass
        
        
class InstrumentalVariableScenario(CausalScenario):
    def __init__(
            self,
            n_samples: int,
            d_c: int,
            d_a: int,
            d_u: int,
            noise_level_treatment: float,
            noise_level_target: float,
            noise_level_instrument: float,
            alpha_corr_covariates: float,
            discrete_binary_instrument: bool = False,
            l_model: str = None,
            m_model: str = None,
            r_model: str = None,
            cv_loss_regression: str = CV_DEFAULT_LOSS_REGRESSION,
            cv_loss_classification: str = CV_DEFAULT_LOSS_CLASSIFICATION,
            specification: str = 'Correct',
            constant_ite: bool = False,
            seed_data: int = 42,
    ):
        super().__init__(
            n_samples,
            d_c,
            d_a,
            noise_level_treatment,
            noise_level_target,
            alpha_corr_covariates,
            l_model,
            m_model,
            cv_loss_regression,
            cv_loss_classification,
            specification,
            constant_ite,
            seed_data
        )
        self.noise_level_instrument = noise_level_instrument
        self.r_model = r_model
        self.U = None
        self.X_c_ex_U = None
        self.X_c_ex_U_and_Z = None
        self.Z = None
        self.discrete_binary_instrument = discrete_binary_instrument
        self.instrument_covariates_relationship_type = "random"

        if d_u > d_c:
            raise ValueError(
                "The number of unobserved covariates must be less or equal to the number of total causal covariates"
            )

    def generate_data(self):
        self._generate_X()

        noise_treatment = np.random.normal(scale=self.noise_level_treatment, size=self.n_samples)
        noise_target = np.random.normal(scale=self.noise_level_target, size=self.n_samples)

        indexes_u = np.random.choice(np.arange(0, self.d_c), size=self.d_u, replace=False)
        self.U = self.X_c[:, indexes_u]
        self.X_c_ex_U = np.delete(self.X_c, indexes_u, axis=1)

        noise_instrument = np.random.normal(scale=self.noise_level_instrument, size=self.n_samples)

        logits = (
            generate_non_linear_relationship(
                self.X_c_ex_U,
                relationship_type=self.instrument_covariates_relationship_type
            )
            + noise_instrument
        )
        if self.discrete_binary_instrument:
            prob_instrument = 1 / (1 + np.exp(-logits))
            self.Z = np.random.binomial(1, p=prob_instrument)
        else:
            self.Z = logits

        self.X_c_ex_U_and_Z = np.column_stack((self.X_c_ex_U, self.Z))
        self.X_c_and_Z = np.column_stack((self.X_c, self.Z))

        logits = (
            generate_non_linear_relationship(
                self.X_c_and_Z,
                relationship_type=self.treatment_covariates_relationship_type
            )
            + noise_treatment
        )
        prob_treatment = 1 / (1 + np.exp(-logits))
        self.treatment = np.random.binomial(1, p=prob_treatment)

        treatment_effect, self.true_ate, self.treatment_covariates_relationship_type = (
            modify_treatment_effect_and_compute_ate(
                treatment=self.treatment,
                relationship_type=self.treatment_covariates_relationship_type,
                X_c=self.X_c,
                constant_ite=False,
                random_seed=self.seed_data
            )
        )

        y0 = generate_non_linear_relationship(
            self.X_c, relationship_type=self.target_covariates_relationship_type
        )
        
        self.target = y0 + treatment_effect + noise_target

    def build_model(self) -> bool:
        if self.target is None:
            raise ValueError('Data has not been generated')
        
        if self.specification == 'Correct':
            X = self.X_c_ex_U
        elif self.specification == 'Inclusion of Non-Causal Cofounders':
            X = np.hstack((self.X_c_ex_U, self.X_a))
        elif self.specification == "Extra Unobserved Confounders":
            n_unobserved = np.ceil(self.d_c * self.pct_unobserved)
            X_c_partial = np.hstack((self.X_c_ex_U, np.random.normal(size=(self.n_samples, n_unobserved))))
            X = np.hstack((X_c_partial, self.X_a))
        elif self.specification == "Instrument treated as Cofounder":
            X = np.hstack((self.X_c_ex_U, self.Z))
        elif self.specification == "Instrument treated as Cofounder and Inclusion of Non-Causal Cofounders":
            X = np.hstack((self.X_c_ex_U, self.Z, self.X_a))
        else:
            raise ValueError(f"Unhandled specification: {self.specification}")

        if self.specification.startswith("Instrument treated as Cofounder"):
            dml_data = DoubleMLData.from_arrays(x=X, y=self.target, t=self.treatment)

            pipeline_l = Pipeline(NORMAL_PIPELINES['regression'][self.l_model])
            pipeline_m = Pipeline(NORMAL_PIPELINES['classification'][self.m_model])
            
            ml_l = add_grid_search(pipeline=pipeline_l, model=self.l_model, scoring=self.cv_loss_regression)
            ml_m = add_grid_search(pipeline=pipeline_m, model=self.m_model, scoring=self.cv_loss_classification)

            self.dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
            return True
        else:
            dml_data = DoubleMLData.from_arrays(x=X, y=self.target, d=self.treatment, z=self.Z)

            instrument_model_type = 'classification' if self.discrete_binary_instrument else 'regression'
            instrument_scoring = self.cv_loss_classification if self.discrete_binary_instrument else self.cv_loss_regression
            
            pipeline_l = Pipeline(NORMAL_PIPELINES['regression'][self.l_model])
            pipeline_m = Pipeline(NORMAL_PIPELINES['classification'][self.m_model])
            pipeline_r = Pipeline(NORMAL_PIPELINES[instrument_model_type][self.r_model])
            
            ml_l = add_grid_search(pipeline=pipeline_l, model=self.l_model, scoring=self.cv_loss_regression)
            ml_m = add_grid_search(pipeline=pipeline_m, model=self.m_model, scoring=self.cv_loss_classification)
            ml_r = add_grid_search(pipeline=pipeline_r, model=self.r_model, scoring=instrument_scoring)
            
            self.dml = DoubleMLPLIV(dml_data, ml_l=ml_l, ml_m=ml_m, ml_r=ml_r, n_folds=5)
            return True
        
    def delete_in_memory(self):
        self.__del__()