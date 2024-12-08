CREATE TABLE IF NOT EXISTS scenario_simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT
    , seed_data INTEGER
    , scenario_name TEXT
    , n_samples INTEGER
    , d_c INTEGER
    , d_a INTEGER
    , d_u INTEGER
    , timestamp_data_generation TEXT      
    , noise_level_treatment REAL
    , noise_level_target REAL
    , noise_level_instrument REAL
    , alpha_corr_covariates REAL
    , true_ate REAL
    , avg_corr_covariates REAL
    , l_model TEXT
    , m_model TEXT
    , r_model TEXT
    , cv_loss_regression TEXT
    , cv_loss_classification TEXT
    , specification TEXT
    , pct_unobserved REAL
    , pct_extra_unobserved REAL
    , binary_instrument BOOLEAN            
    , instrument_covariates_relationship_type TEXT  
    , target_covariates_relationship_type TEXT       
    , treatment_covariates_relationship_type TEXT    
    , estimated_ate REAL
    , std_error REAL
    , ci_2_5_pct REAL
    , ci_97_5_pct REAL
    , estimated_ate_t_stat REAL
    , estimated_ate_p_value REAL
);