import pandas as pd
import numpy as np

from pycox.evaluation import EvalSurv

## Utils function to get transform features and outcomes into right format
def get_features_and_outcomes(data, event_var, dep_var):
    
    X = data.drop([event_var, dep_var], axis=1)
    features = list(X.columns)

    y = np.array(data[[event_var,dep_var]])
    
    # Split up the target into event and outcome
    e = y[: , 0]
    y = y[: , 1]
    
    return X, y, e, features

## In case of DeepHit predictions the predictions are extrapolated 
# and need to be transformed into monthly predictions
def get_discrete_surv_periods(surv):
    # need to interpolate from only 10 discrete period
    surv['Period'] = np.floor(surv.index)
    # aggregate back to discrete periods
    surv = surv.groupby('Period').mean()
    
    return surv


# get the AVG of n stochastic passes through the NN as the MC-Dropout prediction
def mc_dropout_surv_prediction(dropout_model, X, n=10, is_deep_hit = False):

    obs_list = []
    for i in range(n):
        # get stochastic surv. function prediction
        if is_deep_hit:
            # interpolate the binned survival times 
            surv_df_obs = dropout_model.interpolate(10).predict_surv_df(X, eval_ = False)
            surv_df_obs = np.array(get_discrete_surv_periods(surv_df_obs))
        else:
            surv_df_obs = np.array(dropout_model.predict_surv_df(X, eval_ = False))
        
        surv_df_obs = surv_df_obs.reshape( 1 , surv_df_obs.shape[0], surv_df_obs.shape[1])
        obs_list.append(surv_df_obs)
    
    # Mean Prediction
    mean_df = np.mean(obs_list, axis=0)
    mean_df = pd.DataFrame(mean_df[0])
    
    return mean_df        



def get_c_index(model, X, y, is_deep_hit = False):
    if is_deep_hit:
        # interpolate the binned survival times 
        surv = mc_dropout_surv_prediction(model, X, n = 100, is_deep_hit = True)
        surv = get_discrete_surv_periods(surv)
    else:
        surv = mc_dropout_surv_prediction(model, X, n = 100)
        
    ev = EvalSurv(surv, y[0], y[1], censor_surv='km')
    
    return ev.concordance_td()
    
    
## Blumenstock et al. 2020
    
# corrupt one feature at a time and get the score
def get_permutation_results(model, X_test, y_test, n_repeats = 10, is_deep_hit = False):

    num_features = X_test.shape[1]
    # C-Index without corrupting features
    
    permutation_c_index = []

    for feat_idx in range(num_features):

        for i in range(n_repeats):
            # add random noise from N(0,1)
            feature_noise = np.random.normal(0,1, X_test.shape[0])

            X_test_perm = X_test.copy()
            # permute the values of the feature
            X_test_perm[:, feat_idx] = X_test_perm[:, feat_idx] + feature_noise

            c_i_results = []
            # the C-index
            ## Only for DeepHit, interpolation included
            sample_c_index = get_c_index(model, X_test_perm, y_test, is_deep_hit = is_deep_hit)
            
            c_i_results = c_i_results + [sample_c_index]

        # get the average C-Index for the corrupted feature
        feature_score = np.array(c_i_results).mean()

        permutation_c_index = permutation_c_index + [feature_score]

    return permutation_c_index


## 
def transform_aggrgeated_predictions(agg_predictions, pred_colname = 'PRED_MEAN'):
    agg_predictions = agg_predictions.reshape(agg_predictions.shape[1], agg_predictions.shape[2])
    agg_predictions = pd.DataFrame(agg_predictions)
    
    agg_predictions.index = agg_predictions.index + 1
    agg_predictions.columns = agg_predictions.columns + 1
    
    agg_predictions = agg_predictions.reset_index().melt(id_vars='index')
    agg_predictions.columns = ['PERIOD', 'ID', pred_colname]
    
    return agg_predictions

def get_stochastic_surv_func(dropout_model, X, n = 100, model_name='DeepSurv_MC', interpolate =-1):
    obs_list = []

    for i in range(n):
        # get stochastic surv. function prediction
        if interpolate > 0:
            surv_df_obs = dropout_model.interpolate(interpolate).predict_surv_df(X, eval_ = False)
            surv_df_obs = np.array(get_discrete_surv_periods(surv_df_obs))
        else:
            surv_df_obs = np.array(dropout_model.predict_surv_df(X, eval_ = False))
        
        surv_df_obs = surv_df_obs.reshape( 1 , surv_df_obs.shape[0], surv_df_obs.shape[1])
        obs_list.append(surv_df_obs)
    
    # Mean Prediction
    mean_df = np.mean(obs_list, axis=0)
    #if len(mean_df.shape) == 3:
    #    mean_df = mean_df[0]
    mean_df = transform_aggrgeated_predictions(mean_df, 'PRED_MEAN')
    
    ## 95% percentile
    percentile = 95
    
    # Percentiles as the CI bounds    
    lower_percentile_bound = (100 - percentile) / 2.
    upper_percentile_bound = 100 - lower_percentile_bound
    
    lower_95_bound_df = np.percentile(obs_list, lower_percentile_bound, axis=0)
    lower_95_bound_df = transform_aggrgeated_predictions(lower_95_bound_df, 'PRED_95_LOW')
    
    upper_95_bound_df = np.percentile(obs_list, upper_percentile_bound, axis=0)
    upper_95_bound_df = transform_aggrgeated_predictions(upper_95_bound_df, 'PRED_95_HIGH')
    
    ## 80% percentile
    percentile = 80
    
    # Percentiles as the CI bounds    
    lower_percentile_bound = (100 - percentile) / 2.
    upper_percentile_bound = 100 - lower_percentile_bound
    
    lower_80_bound_df = np.percentile(obs_list, lower_percentile_bound, axis=0)
    lower_80_bound_df = transform_aggrgeated_predictions(lower_80_bound_df, 'PRED_80_LOW')
    
    upper_80_bound_df = np.percentile(obs_list, upper_percentile_bound, axis=0)
    upper_80_bound_df = transform_aggrgeated_predictions(upper_80_bound_df, 'PRED_80_HIGH')
    
    # merge the predictions as one DF
    full_prediction = pd.merge(mean_df, lower_95_bound_df, how='left', on = ['ID','PERIOD'])
    full_prediction = pd.merge(full_prediction, upper_95_bound_df, how='left', on = ['ID','PERIOD'])
    full_prediction = pd.merge(full_prediction, lower_80_bound_df, how='left', on = ['ID','PERIOD'])
    full_prediction = pd.merge(full_prediction, upper_80_bound_df, how='left', on = ['ID','PERIOD'])
    
    # always let the periods start at 1
    if full_prediction['PERIOD'].min() == 0:
        full_prediction['PERIOD'] = full_prediction['PERIOD'] + 1
    
    full_prediction['MODEL'] = model_name
    
    # rearrange columns 
    full_prediction = full_prediction[['ID','PERIOD','MODEL','PRED_80_LOW','PRED_80_HIGH','PRED_95_LOW','PRED_95_HIGH', 'PRED_MEAN']]
    
    return full_prediction