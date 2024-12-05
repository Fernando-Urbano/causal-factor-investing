from doubleml.double_ml import DoubleML
from doubleml.double_ml_data import DoubleMLData
from doubleml.utils import (_dml_cv_predict, _dml_tune, _check_finite_predictions,
                            _check_is_propensity, _check_score)
from sklearn.base import clone
import numpy as np
import pandas as pd

class DoubleMLFrontdoor(DoubleML):
    """Double machine learning implementation of the frontdoor adjustment."""

    def __init__(self, obj_dml_data, ml_m, ml_h, n_folds=5, n_rep=1, score='frontdoor', draw_sample_splitting=True):
        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)

        # Data checks
        self._check_data(self._dml_data)
        _check_score(self.score, valid_scores=['frontdoor'], allow_callable=True)

        # Initialize learners
        self._learner = {'ml_m': clone(ml_m), 'ml_h': clone(ml_h)}

        # Check learners
        ml_m_is_classifier = self._check_learner(self._learner['ml_m'], 'ml_m', regressor=True, classifier=True)
        ml_h_is_classifier = self._check_learner(self._learner['ml_h'], 'ml_h', regressor=True, classifier=True)

        # Set prediction methods
        self._predict_method = {}
        if ml_m_is_classifier:
            self._predict_method['ml_m'] = 'predict_proba'
        else:
            self._predict_method['ml_m'] = 'predict'

        if ml_h_is_classifier:
            self._predict_method['ml_h'] = 'predict_proba'
        else:
            self._predict_method['ml_h'] = 'predict'

        # Initialize nuisance parameters
        self._initialize_ml_nuisance_params()

        # Flags
        self._sensitivity_implemented = False  # Update if sensitivity analysis is implemented
        self._external_predictions_implemented = True

        # Define score element names
        self._score_element_names = ['psi_a', 'psi_b']

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'Got {type(obj_dml_data)}.')
        # Additional data checks can be added here
        # For example, check that 'M' is in x_cols
        if 'M' not in obj_dml_data.x_cols:
            raise ValueError("Mediator 'M' must be included in x_cols.")

    def _initialize_ml_nuisance_params(self):
        self._params = {'ml_m': {key: [None] * self.n_rep for key in self._dml_data.d_cols},
                        'ml_h': {key: [None] * self.n_rep for key in self._dml_data.d_cols}}

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x = self._dml_data.x
        y = self._dml_data.y
        d = self._dml_data.d

        m_external = external_predictions.get('ml_m') is not None
        h_external = external_predictions.get('ml_h') is not None

        # Nuisance ml_m (Outcome model m(M_i, C_i))
        if m_external:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'),
                                    method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)

        # Nuisance ml_h (Mediator model h(T_i, C_i))
        if h_external:
            h_hat = {'preds': external_predictions['ml_h'],
                     'targets': None,
                     'models': None}
        else:
            # Prepare data for ml_h
            x_h = pd.concat([d.reset_index(drop=True), x.drop(columns='M').reset_index(drop=True)], axis=1)
            y_h = x['M'].reset_index(drop=True)
            h_hat = _dml_cv_predict(self._learner['ml_h'], x_h, y_h, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_h'),
                                    method=self._predict_method['ml_h'],
                                    return_models=return_models)
            _check_finite_predictions(h_hat['preds'], self._learner['ml_h'], 'ml_h', smpls)

        # Predict h(1, C_i) and h(0, C_i)
        h1_hat = np.full_like(y, np.nan)
        h0_hat = np.full_like(y, np.nan)
        for idx, (train_idx, test_idx) in enumerate(smpls):
            # Access fitted model for fold
            if return_models:
                ml_h_model = h_hat['models'][idx]
            else:
                ml_h_model = clone(self._learner['ml_h'])
                ml_h_model.fit(x_h.iloc[train_idx], y_h[train_idx])

            # Prepare data for T=1 and T=0
            C_test = x.drop(columns='M').iloc[test_idx].reset_index(drop=True)
            T1 = pd.Series(np.ones(len(C_test)), name='T')
            T0 = pd.Series(np.zeros(len(C_test)), name='T')
            X_h1 = pd.concat([T1, C_test], axis=1)
            X_h0 = pd.concat([T0, C_test], axis=1)

            h1_preds = ml_h_model.predict(X_h1)
            h0_preds = ml_h_model.predict(X_h0)

            h1_hat[test_idx] = h1_preds
            h0_hat[test_idx] = h0_preds

        # Compute score elements
        psi_a, psi_b = self._score_elements(y, d, m_hat['preds'], h_hat['preds'], h1_hat, h0_hat)
        psi_elements = {'psi_a': psi_a, 'psi_b': psi_b}

        # Prepare predictions and targets
        preds = {'predictions': {'ml_m': m_hat['preds'],
                                 'ml_h': h_hat['preds']},
                 'targets': {'ml_m': y,
                             'ml_h': x['M']},
                 'models': {'ml_m': m_hat.get('models'),
                            'ml_h': h_hat.get('models')}}

        return psi_elements, preds

    def _score_elements(self, y, d, m_hat, h_hat, h1_hat, h0_hat):
        # Compute necessary components
        M = self._dml_data.data['M'].values
        e_M = h_hat * (1 - h_hat)
        epsilon = 1e-6
        e_M = np.maximum(e_M, epsilon)

        term1 = (M - h_hat) / e_M
        term2 = y - m_hat
        delta_h = h1_hat - h0_hat

        psi_a = term1 * term2 * delta_h
        psi_b = np.ones_like(psi_a)

        return psi_a, psi_b

    def _est_coef(self, psi_elements, smpls=None, scaling_factor=None, inds=None):
        # Estimate the causal parameter theta
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        theta_hat = -np.mean(psi_b) / np.mean(psi_a)
        return theta_hat

    def _compute_score(self, psi_elements, coef):
        # Compute the score function
        psi = psi_elements['psi_a'] * coef + psi_elements['psi_b']
        return psi

    def _compute_score_deriv(self, psi_elements, coef):
        # Compute the derivative of the score function with respect to theta
        psi_deriv = psi_elements['psi_a']
        return psi_deriv

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        # Implement hyperparameter tuning using _dml_tune
        x = self._dml_data.x
        y = self._dml_data.y
        d = self._dml_data.d

        if scoring_methods is None:
            scoring_methods = {'ml_m': None,
                               'ml_h': None}

        train_inds = [train_index for (train_index, _) in smpls]

        # Tuning ml_m
        m_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        # Tuning ml_h
        x_h = pd.concat([d.reset_index(drop=True), x.drop(columns='M').reset_index(drop=True)], axis=1)
        y_h = x['M'].reset_index(drop=True)
        h_tune_res = _dml_tune(y_h, x_h, train_inds,
                               self._learner['ml_h'], param_grids['ml_h'], scoring_methods['ml_h'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_best_params = [res.best_params_ for res in m_tune_res]
        h_best_params = [res.best_params_ for res in h_tune_res]

        params = {'ml_m': m_best_params,
                  'ml_h': h_best_params}
        tune_res = {'m_tune': m_tune_res,
                    'h_tune': h_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']
