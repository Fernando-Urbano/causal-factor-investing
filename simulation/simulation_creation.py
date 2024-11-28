import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
import datetime
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


NON_LINEAR_TRANSFORMATIONS = {
    "linear": lambda t: t,
    "sin": lambda t: np.sin(t),
    "cos": lambda t: np.cos(t),
    "square": lambda t: t**2,
    "cubic": lambda t: t**3,
    "step": lambda t: np.where(t > 0.5, 2, -1),
    "piecewise_linear": lambda t: np.piecewise(
        t, [t < 0.33, (t >= 0.33) & (t < 0.66), t >= 0.66],
        [lambda x: x * 2, lambda x: x + 1, lambda x: x * 0.5]
    ),
    "exponential": lambda t: np.exp(t / 5),
    "logarithmic": lambda t: np.log1p(np.abs(t)),
}


PARAM_GRID = {
    "RandomForest": {
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
        'model__learning_rate': ['constant', 'adaptive']
    }
}

PIPELINES_DNN_LASSO = {"Lasso": [
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5, random_state=42))
    ],
    "DNN": [
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(max_iter=1000, random_state=42, activation='relu'))
    ]
}

NORMAL_PIPELINES = {
    "classification": {
        PIPELINES_DNN_LASSO.update({
            "RandomForestClassifier": [('model', RandomForestClassifier(random_state=42))]
        })
    },
    "regression": {
        PIPELINES_DNN_LASSO.update({
            "RandomForestRegressor": [('model', RandomForestRegressor(random_state=42))],
        })
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


def covariance_to_correlation(cov_matrix):
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
    corr_matrix_wout_diag = np.fill_diagonal(corr_matrix, np.nan)
    return corr_matrix_wout_diag.mean().mean()


def generate_non_linear_relationship(X, relationship_type="random", random_seed=None):
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
    if random_seed is not None:
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

    weights = np.random.uniform(-1, 1, size=n_features)
    y = transformed_features @ weights

    return y


def modify_treatment_effect_and_compute_ate(treatment, relationship_type="random", random_seed=None):
    """
    Modifies the treatment effect with a variety of non-linear relationships and computes the true ATE.

    Parameters:
    - treatment (np.ndarray): Array of treatment assignments (0 or 1).
    - X (np.ndarray): Covariate matrix (n_samples x n_features).
    - relationship_type (str): Type of non-linear relationship. Options:
        - "sin"
        - "cos"
        - "square"
        - "cubic"
        - "step" (piecewise constant)
        - "piecewise_linear"
        - "interaction" (interacts with a covariate)
        - "exponential"
        - "logarithmic"
        - "random" (randomly selects one of the above)
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - treatment_effect (np.ndarray): Non-linear treatment effect.
    - true_ate (float): True average treatment effect.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if relationship_type == "random":
        relationship_type = np.random.choice(list(NON_LINEAR_TRANSFORMATIONS.keys()))

    if relationship_type not in NON_LINEAR_TRANSFORMATIONS:
        raise ValueError(
            f"Invalid relationship_type '{relationship_type}'."
            + f" Must be one of {list(NON_LINEAR_TRANSFORMATIONS.keys())}."
        )

    non_linear_function = NON_LINEAR_TRANSFORMATIONS[relationship_type]
    treatment_effect = NON_LINEAR_TRANSFORMATIONS(treatment)

    if relationship_type == "interaction":
        true_ate = np.mean(non_linear_function(1) - non_linear_function(0))
    else:
        true_ate = np.mean(non_linear_function(1) - non_linear_function(0))

    return treatment_effect, true_ate, relationship_type


class CausalScenario:
    def __init__(
            self,
            n_samples: int, d_c: int, d_a: int,
            noise_level: float,
            alpha_corr_covariates: float,
            model: str = None,
            cv_loss_regression: str = CV_DEFAULT_LOSS_REGRESSION,
            cv_loss_classification: str = CV_DEFAULT_LOSS_CLASSIFICATION,
            specification: str = 'correct'
        ):
        self.n_samples = n_samples
        self.d_c = d_c
        self.d_a = d_a
        self.timestamp_data_generation = None
        self.noise_level = noise_level
        self.alpha_corr_covariates = alpha_corr_covariates
        self.true_ate = None
        self.X_c = None
        self.X_a = None
        self.target = None
        self.treatment = None
        self.avg_corr_covariates = None
        self.model = model
        self.cv_loss_regression = cv_loss_regression
        self.cv_loss_classification = cv_loss_classification
        self.specification = specification
        self.treatment_target_relationship_type = "random"
        self.target_covariates_relationship_type = "random"
        self.summary = None

    def generate_data(self):
        self._set_generate_data_timestamp()
        d = self.d_c + self.d_a
        
        cov_matrix = make_sparse_spd_matrix(d, alpha=self.alpha_corr_covariates) * make_spd_matrix(d)
        self.avg_corr_covariates = calc_avg_corr(cov_matrix)
        X = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_matrix, size=self.n_samples)
        self.X_c = X[:, :self.d_c]
        self.X_a = X[:, -self.d_a:]
        assert X.size[1] == self.X_c.size[1] + self.X_c.size[1], "Invalid sizes for X_a and X_c"

        logits = -generate_non_linear_relationship(self.X_c)
        self.treatment = np.random.binomial(1, p=1 / (1 + np.exp(logits)))

        noise = np.random.normal(scale=self.noise_level, size=self.n_samples)
        treatment_effect, self.true_ate, self.treatment_target_relationship_type = (
            modify_treatment_effect_and_compute_ate(self.treatment, self.treatment_target_relationship_type)
        )
        self.target = (
            generate_non_linear_relationship(self.X_c, relationship_type=self.target_covariates_relationship_type)
            + treatment_effect + noise
        )

    def set_specification(self, new_specification):
        self.specification = new_specification

    def set_treatment_target_relationship_type(self, new_relationship):
        self.treatment_target_relationship_type = new_relationship

    def build_model(self):
        pass

    def _set_generate_data_timestamp(self) -> None:
        self.timestamp_data_generation = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def _build_summary(self) -> bool:
        if self.dml_plr is None:
            raise ValueError('dml_plr is None')
        
        summary = self.dml_plr.summary
        summary.reset_index(drop=True, inplace=True)
        summary.rename({
            'coef': 'estimated_ate', 'std err': 'std_error',
            '2.5 %': 'ci_2_5_pct', '97.5 %': 'ci_97_5_pct'
        }, axis=1, inplace=True)
        model_information = pd.DataFrame({
            "n_samples": self.n_samples,
            "d_c": self.d_c,
            "d_a": self.d_a,
            "timestamp_data_generation": self.timestamp_data_generation,
            "noise_level": self.noise_level,
            "alpha_corr_covariates": self.alpha_corr_covariates,
            "true_ate": self.true_ate,
            "avg_corr_covariates": self.avg_corr_covariates,
            "model": self.model,
            "cv_loss_regression ": self.cv_loss_regression ,
            "cv_loss_classification": self.cv_loss_classification,
            "specification": self.specification,
            "target_covariates_relationship_type": self.target_covariates_relationship_type,
            "treatment_target_relationship_type": self.treatment_target_relationship_type,
        }, index=[0])
        self.summary = pd.concat([summary, model_information], axis=1)

    def get_summary(self) -> pd.DataFrame:
        self._build_summary()
        return self.summary
    
    def save_summary(self) -> bool:
        self._build_summary()
        # TODO
        return True


class BackdoorAdjustmentScenario(CausalScenario):
    def __init__(self, n_samples, d_c, d_a, noise_level, alpha_corr_covariates):
        super().__init__(n_samples, d_c, d_a, noise_level, alpha_corr_covariates)

    def build_model(self) -> bool:
        if self.target is None:
            raise ValueError('Data has not been generated')
        
        if self.specification == 'correct':
            X = self.X_c
        elif self.specification == 'all features':
            X = np.hstack((self.X_c, self.X_a))
        elif self.specification == 'no feature':
            X = np.matrix()
        dml_data = DoubleMLData.from_arrays(x=X, y=self.target, t=self.treatment)

        pipeline_l = Pipeline(NORMAL_PIPELINES['regression'][self.model])
        pipeline_g = Pipeline(NORMAL_PIPELINES['classification'][self.model])

        ml_l = add_grid_search(pipeline=pipeline_l, model=self.model, scoring=self.cv_loss_regression)
        ml_g = add_grid_search(pipeline=pipeline_g, model=self.model, scoring=self.cv_loss_classification)

        self.dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_g, n_folds=5)

        return True

    def fit_model(self) -> bool:
        self.dml_plr.fit()
        return True
        

class FrontdoorAdjustmentScenario(CausalScenario):
    def __init__(self, n_samples, d_c, d_a, noise_level, alpha_corr_covariates):
        super().__init__(n_samples, d_c, d_a, noise_level, alpha_corr_covariates)

    def generate_data(self):
        pass
        
        
class InstrumentalVariableScenario(CausalScenario):
    def __init__(self, n_samples, d_c, d_a, noise_level, alpha_corr_covariates):
        super().__init__(n_samples, d_c, d_a, noise_level, alpha_corr_covariates)

    def generate_data(self):
        pass