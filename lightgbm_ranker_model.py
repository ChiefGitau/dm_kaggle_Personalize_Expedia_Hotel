import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import optuna

RANDOM_STATE = 42 # Consistent random state

def perform_cross_validation(X, y, groups_for_splitting, df_full_for_group_counts, n_folds=5, lgbm_params=None):
    """
    Performs GroupKFold cross-validation for an LGBMRanker.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target relevance scores.
        groups_for_splitting (pd.Series): Series of group IDs (e.g., srch_id) for each row in X, used for splitting.
        df_full_for_group_counts (pd.DataFrame): The full DataFrame (or sample) from which X and y were derived,
                                                 must contain the group ID column (e.g., \'srch_id\')
                                                 to correctly calculate group sizes for subsets.
        n_folds (int): Number of folds for GroupKFold.
        lgbm_params (dict, optional): Parameters for LGBMRanker. Defaults to basic params if None.

    Returns:
        tuple: (mean_ndcg_score, std_ndcg_score, mean_feature_importances_df)
    """
    print(f"\\n--- Performing {n_folds}-Fold Cross-Validation ---")
    gkf = GroupKFold(n_splits=n_folds)
    fold_ndcg_scores = []
    all_feature_importances = pd.DataFrame()

    default_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'label_gain': [0, 1, 5], # Assuming relevance 0, 1, 2 maps to gains 0, 1, 5
        'eval_at': [5],
        'n_estimators': 100,
        'learning_rate': 0.1,
        'importance_type': 'gain',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': -1
    }
    current_lgbm_params = default_params.copy()
    if lgbm_params:
        current_lgbm_params.update(lgbm_params)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups_for_splitting)):
        print(f"--- Fold {fold + 1}/{n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Group counts for the current fold
        train_groups = df_full_for_group_counts.iloc[train_idx].groupby('srch_id').size().to_numpy()
        val_groups = df_full_for_group_counts.iloc[val_idx].groupby('srch_id').size().to_numpy()

        if len(train_groups) == 0 or len(val_groups) == 0 or X_train.empty or X_val.empty:
            print(f"Skipping fold {fold + 1} due to empty train or validation groups/data.")
            continue
        
        # Ensure eval_metric is set correctly for callbacks if present in params
        eval_metric_cb = 'ndcg'
        if 'metric' in current_lgbm_params:
            if isinstance(current_lgbm_params['metric'], list):
                eval_metric_cb = current_lgbm_params['metric'][0] # Take the first one for early stopping
            else:
                eval_metric_cb = current_lgbm_params['metric']


        ranker_cv = lgb.LGBMRanker(**current_lgbm_params)
        ranker_cv.fit(
            X_train, y_train, group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_metric=eval_metric_cb, # metric for early stopping
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )

        if ranker_cv.evals_result_ and 'valid_0' in ranker_cv.evals_result_ and f'{eval_metric_cb}@5' in ranker_cv.evals_result_['valid_0']:
            ndcg_at_5 = ranker_cv.evals_result_['valid_0'][f'{eval_metric_cb}@5'][-1]
            fold_ndcg_scores.append(ndcg_at_5)
            print(f"Fold {fold + 1} NDCG@5: {ndcg_at_5:.4f}")

            fold_importances_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': ranker_cv.feature_importances_,
                'fold': fold + 1
            })
            all_feature_importances = pd.concat([all_feature_importances, fold_importances_df], ignore_index=True)
        else:
            print(f"Could not retrieve NDCG@5 for fold {fold + 1}. Check eval_metric and results structure.")
            # print(ranker_cv.evals_result_) # for debugging

    mean_ndcg = np.mean(fold_ndcg_scores) if fold_ndcg_scores else 0.0
    std_ndcg = np.std(fold_ndcg_scores) if fold_ndcg_scores else 0.0
    mean_importances = all_feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False) if not all_feature_importances.empty else pd.Series()
    
    print(f"Mean NDCG@5 across {len(fold_ndcg_scores)} folds: {mean_ndcg:.4f} +/- {std_ndcg:.4f}")
    return mean_ndcg, std_ndcg, mean_importances


def tune_hyperparameters_optuna(X, y, groups_for_splitting, df_full_for_group_counts, n_trials=20, n_cv_folds=3):
    """
    Tunes LGBMRanker hyperparameters using Optuna.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target relevance scores.
        groups_for_splitting (pd.Series): Group IDs for each row in X.
        df_full_for_group_counts (pd.DataFrame): The full DataFrame (or sample) containing 'srch_id'.
        n_trials (int): Number of Optuna trials.
        n_cv_folds (int): Number of CV folds within each Optuna trial.

    Returns:
        dict: Best hyperparameters found.
    """
    print(f"\\n--- Tuning Hyperparameters with Optuna ({n_trials} trials, {n_cv_folds} CV folds each) ---")

    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'label_gain': [0, 1, 5],
            'eval_at': [5],
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 700, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        
        # Using a local variable for clarity within the objective function
        current_X, current_y = X, y 
        current_groups_for_splitting_objective = groups_for_splitting

        gkf_optuna = GroupKFold(n_splits=n_cv_folds)
        fold_scores = []

        for fold_num, (train_idx, val_idx) in enumerate(gkf_optuna.split(current_X, current_y, groups=current_groups_for_splitting_objective)):
            X_train_trial, X_val_trial = current_X.iloc[train_idx], current_X.iloc[val_idx]
            y_train_trial, y_val_trial = current_y.iloc[train_idx], current_y.iloc[val_idx]

            train_groups_trial = df_full_for_group_counts.iloc[train_idx].groupby('srch_id').size().to_numpy()
            val_groups_trial = df_full_for_group_counts.iloc[val_idx].groupby('srch_id').size().to_numpy()

            if len(train_groups_trial) == 0 or len(val_groups_trial) == 0 or X_train_trial.empty or X_val_trial.empty:
                # print(f"Optuna trial {trial.number}, fold {fold_num+1}: Skipping due to empty data/groups.")
                return -1.0 # Penalize this trial heavily

            model_trial = lgb.LGBMRanker(**params)
            model_trial.fit(
                X_train_trial, y_train_trial, group=train_groups_trial,
                eval_set=[(X_val_trial, y_val_trial)],
                eval_group=[val_groups_trial],
                eval_metric='ndcg',
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
            
            if model_trial.evals_result_ and 'valid_0' in model_trial.evals_result_ and 'ndcg@5' in model_trial.evals_result_['valid_0']:
                score = model_trial.evals_result_['valid_0']['ndcg@5'][-1]
                fold_scores.append(score)
            else:
                # print(f"Optuna trial {trial.number}, fold {fold_num+1}: NDCG@5 not found.")
                fold_scores.append(0.0)
        
        avg_score = np.mean(fold_scores) if fold_scores else 0.0
        # print(f"Trial {trial.number} avg NDCG@5: {avg_score:.4f} for params: {trial.params}")
        return avg_score

    study = optuna.create_study(direction='maximize', study_name='lgbm_ranker_tuning')
    # optuna.logging.set_verbosity(optuna.logging.INFO) # Control Optuna's output level

    try:
        study.optimize(objective, n_trials=n_trials, timeout=None) # Add timeout in seconds if desired
        print(f"Optuna study finished. Best trial NDCG@5: {study.best_value:.4f}")
        print("Best parameters:", study.best_params)
        return study.best_params
    except Exception as e:
        print(f"Error during Optuna optimization: {e}")
        return {} # Return empty dict or default params on error

def train_final_model(X_train_full, y_train_full, groups_train_full, df_full_for_group_counts, best_params):
    """
    Trains the final LGBMRanker model on the full (sampled) training data.

    Args:
        X_train_full (pd.DataFrame): Full training feature DataFrame.
        y_train_full (pd.Series): Full training target relevance scores.
        groups_train_full (np.array): Group sizes for the full training data.
        df_full_for_group_counts (pd.DataFrame): The full DataFrame used for group splitting in early stopping.
        best_params (dict): Best hyperparameters from tuning.

    Returns:
        lgb.LGBMRanker: Trained model.
    """
    print("\\n--- Training Final Model ---")
    final_model_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'label_gain': [0, 1, 5],
        'eval_at': [5],
        'importance_type': 'gain',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': -1
    }
    final_model_params.update(best_params)
    if 'n_estimators' not in final_model_params or final_model_params.get('n_estimators', 0) < 50 : # Ensure n_estimators is reasonable
        final_model_params['n_estimators'] = 300 # Default if not properly set by tuning
    
    print("Final model parameters for training:")
    print(final_model_params)

    final_model = lgb.LGBMRanker(**final_model_params)

    # Optional: Use a small validation split from the training data for early stopping
    # This is generally good practice.
    temp_df_for_final_split = pd.DataFrame({
        'srch_id': df_full_for_group_counts['srch_id'], # Use the group ID from the original df
        'original_index': X_train_full.index # Map back to X_train_full
    }).drop_duplicates(subset=['srch_id'])


    # Make sure we only use indices that are present in X_train_full
    valid_indices_for_split = temp_df_for_final_split[temp_df_for_final_split['original_index'].isin(X_train_full.index)]
    
    if len(valid_indices_for_split) > 10 : # Ensure enough groups for a split
        final_train_srch_ids, final_val_srch_ids = np.split(
            valid_indices_for_split['srch_id'].sample(frac=1, random_state=RANDOM_STATE),
            [int(0.9 * len(valid_indices_for_split))]
        )
        
        final_train_indices = X_train_full[df_full_for_group_counts.loc[X_train_full.index, 'srch_id'].isin(final_train_srch_ids)].index
        final_val_indices = X_train_full[df_full_for_group_counts.loc[X_train_full.index, 'srch_id'].isin(final_val_srch_ids)].index

        if not final_train_indices.empty and not final_val_indices.empty:
            X_fit, X_eval_stop = X_train_full.loc[final_train_indices], X_train_full.loc[final_val_indices]
            y_fit, y_eval_stop = y_train_full.loc[final_train_indices], y_train_full.loc[final_val_indices]
            
            groups_fit = df_full_for_group_counts.loc[final_train_indices].groupby('srch_id').size().to_numpy()
            groups_eval_stop = df_full_for_group_counts.loc[final_val_indices].groupby('srch_id').size().to_numpy()

            if len(groups_fit) > 0 and len(groups_eval_stop) > 0:
                print(f"Fitting final model with early stopping on 90/10 split of training data.")
                final_model.fit(
                    X_fit, y_fit, group=groups_fit,
                    eval_set=[(X_eval_stop, y_eval_stop)],
                    eval_group=[groups_eval_stop],
                    eval_metric='ndcg',
                    callbacks=[lgb.early_stopping(10, verbose=1)]
                )
            else: # Fallback if eval split is problematic
                print("Early stopping validation set for final model is problematic, fitting on all provided training data without early stopping.")
                final_model.fit(X_train_full, y_train_full, group=groups_train_full)
        else: # Fallback if indices are empty
            print("Train/Val indices for final model early stopping are empty. Fitting on all provided training data without early stopping.")
            final_model.fit(X_train_full, y_train_full, group=groups_train_full)

    else: # Fallback if not enough groups
        print("Not enough groups for early stopping split in final model training. Fitting on all provided training data.")
        final_model.fit(X_train_full, y_train_full, group=groups_train_full)
        
    print("Final model training completed.")
    return final_model

def predict_and_format_submission(model, X_test, df_test_original_ids, submission_filename="submission.csv"):
    """
    Makes predictions on the test set and formats for Kaggle submission.

    Args:
        model (lgb.LGBMRanker): Trained LGBMRanker model.
        X_test (pd.DataFrame): Preprocessed test feature DataFrame.
        df_test_original_ids (pd.DataFrame): Test DataFrame containing 'srch_id' and 'prop_id'.
        submission_filename (str): Name for the output submission CSV file.
    """
    print("\\n--- Predicting on Test Data and Formatting Submission ---")
    if X_test.empty:
        print("X_test is empty. Cannot make predictions.")
        return

    test_predictions = model.predict(X_test)
    
    # Add predictions to a copy of the test DataFrame that has original IDs
    df_test_with_preds = df_test_original_ids.copy()
    # Ensure index alignment if X_test was derived from df_test_original_ids and then possibly subsetted/reordered.
    # If X_test maintains original index from df_test_original_ids:
    if X_test.index.equals(df_test_with_preds.index):
         df_test_with_preds['predicted_score'] = test_predictions
    else:
        # If indices are not aligned, one must be careful. Assuming X_test rows correspond to df_test_original_ids rows.
        # This part might need adjustment based on how X_test was constructed.
        # For simplicity, if they don't match, we'll assume direct assignment is okay if lengths match.
        # A safer way is to ensure X_test keeps original indices or merge predictions back carefully.
        print("Warning: X_test index does not match df_test_original_ids index. Assuming row order is preserved for predictions.")
        if len(X_test) == len(df_test_with_preds):
            df_test_with_preds['predicted_score'] = test_predictions
        else:
            print(f"Error: Length mismatch between X_test ({len(X_test)}) and df_test_original_ids ({len(df_test_with_preds)}). Cannot assign predictions.")
            return


    submission_list = []
    for srch_id, group_df in df_test_with_preds.groupby('srch_id'):
        ranked_properties = group_df.sort_values('predicted_score', ascending=False)
        for _, row in ranked_properties.iterrows():
            submission_list.append({'srch_id': int(row['srch_id']), 'prop_id': int(row['prop_id'])})
    
    df_submission = pd.DataFrame(submission_list)
    df_submission.to_csv(submission_filename, index=False)
    print(f"Submission file '{submission_filename}' created. Top 5 rows:")
    print(df_submission.head()) 