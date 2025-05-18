import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class ContextualBanditRanker:

    
    def __init__(self, alpha=0.2, exploration_strategy='ucb', base_algorithm='rf',
                 n_estimators=100, learning_rate=0.01, max_depth=5):

        self.alpha = alpha
        self.exploration_strategy = exploration_strategy
        self.base_algorithm = base_algorithm
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reward_models = {}  # Models for each action (hotel property)
        self.action_counts = {}  # Count of each action being chosen
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # For storing uncertainty estimates
        self.uncertainty_estimates = {}
        
        # Initialize the base algorithm
        if self.base_algorithm == 'rf':
            self.base_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif self.base_algorithm == 'linear':
            self.base_model = LogisticRegression(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        
    def _create_context_reward_pairs(self, X, y, groups):

        contexts = []
        actions = []
        rewards = []
        
        # Get unique group IDs
        unique_groups = groups.unique()
        
        for group_id in unique_groups:
            # Get indices for this group
            group_mask = groups == group_id
            group_X = X[group_mask]
            group_y = y[group_mask]
            
            if len(group_X) == 0 or len(group_y) == 0:
                continue  # Skip empty groups
                
            # For each context (search), we have multiple actions (hotels)
            try:
                context_features = group_X.iloc[0].copy()
            except IndexError:
                continue  # Skip if we can't get the first row
            
            # Extract hotel-specific features
            hotel_features = ['prop_starrating', 'prop_review_score', 'prop_brand_bool',
                            'prop_location_score1', 'prop_location_score2', 
                            'prop_log_historical_price', 'price_usd']
            
            hotel_features = [f for f in hotel_features if f in X.columns]
            
            # Process each hotel for this search
            for idx in range(len(group_X)):
                row = group_X.iloc[idx]
                action_features = row[hotel_features].values
                
                # The action is represented by the combination of hotel features
                action = tuple(action_features)
                
                # The reward is the relevance score (0, 1, or 2)
                reward = group_y.iloc[idx]
                
                # Store the context, action, and reward
                contexts.append(context_features.values)
                actions.append(action)
                rewards.append(reward)
        
        if not contexts:  # Check if we have any valid context-reward pairs
            raise ValueError("No valid context-reward pairs could be created. Check your data.")
            
        return np.array(contexts), actions, np.array(rewards)
    
    def _exploration_bonus(self, action, count):

        if self.exploration_strategy == 'ucb':
            # Upper Confidence Bound
            if count == 0:
                return 10.0  # High value for unexplored actions
            return self.alpha * np.sqrt(np.log(self.total_count + 1) / count)
        
        elif self.exploration_strategy == 'thompson':
            # Thompson Sampling (using model-based uncertainty)
            if action in self.uncertainty_estimates:
                return self.alpha * self.uncertainty_estimates[action]
            return self.alpha * 1.0  # Default uncertainty
        
        elif self.exploration_strategy == 'epsilon_greedy':
            # Epsilon-greedy (random exploration with probability alpha)
            if np.random.random() < self.alpha:
                return np.random.random()  # Random bonus
            return 0.0
            
        return 0.0
    
    def fit(self, X, y, groups):

        self.feature_names = X.columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Create context-reward pairs
        try:
            contexts, actions, rewards = self._create_context_reward_pairs(X_scaled, y, groups)
        except ValueError as e:
            print(f"Error in creating context-reward pairs: {e}")
            return self
        
        # Initialize counts
        unique_actions = set(actions)
        self.action_counts = {action: 0 for action in unique_actions}
        self.total_count = 0
        
        # Initialize reward models for each action
        self.reward_models = {}
        
        # Train a model to predict rewards based on contexts and actions
        combined_features = []
        combined_rewards = []
        
        for i, (context, action, reward) in enumerate(zip(contexts, actions, rewards)):
            # Combine context and action features
            combined_feature = np.concatenate([context, action])
            combined_features.append(combined_feature)
            combined_rewards.append(reward)
            
            # Update counts
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.total_count += 1
        
        # Train a single model on all combined features
        X_train = np.array(combined_features)
        y_train = np.array(combined_rewards)
        
        if self.base_algorithm == 'rf':
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
        model.fit(X_train, y_train)
        self.global_model = model
        
        # Store uncertainty estimates for Thompson sampling
        if self.exploration_strategy == 'thompson' and self.base_algorithm == 'rf':
            for action in unique_actions:
                action_indices = [i for i, a in enumerate(actions) if a == action]
                if action_indices:
                    action_features = [combined_features[i] for i in action_indices]
                    predictions = np.array([tree.predict(action_features) for tree in model.estimators_])
                    self.uncertainty_estimates[action] = np.mean(np.std(predictions, axis=0))
        
        self.is_fitted = True
        return self
    
    def predict(self, X, return_scores=False):

        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Create context-action pairs for prediction
        contexts = []
        actions = []
        original_indices = []
        
        # Extract hotel-specific features
        hotel_features = ['prop_starrating', 'prop_review_score', 'prop_brand_bool',
                         'prop_location_score1', 'prop_location_score2', 
                         'prop_log_historical_price', 'price_usd']
        
        hotel_features = [f for f in hotel_features if f in X.columns]
        
        for idx, row in X_scaled.iterrows():
            context_features = row.copy()
            action_features = row[hotel_features].values
            action = tuple(action_features)
            
            # Combine context and action features
            combined_feature = np.concatenate([context_features.values, action_features])
            contexts.append(combined_feature)
            actions.append(action)
            original_indices.append(idx)
        
        # Make predictions using the global model
        X_pred = np.array(contexts)
        predicted_rewards = self.global_model.predict(X_pred)
        
        # Add exploration bonus
        if self.exploration_strategy != 'none':
            for i, action in enumerate(actions):
                count = self.action_counts.get(action, 0)
                bonus = self._exploration_bonus(action, count)
                predicted_rewards[i] += bonus
        
        if return_scores:
            return predicted_rewards
        
        # Return indices sorted by predicted reward (descending)
        return np.argsort(-predicted_rewards)
    
    def evaluate(self, X, y, groups, at_k=5):

        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        ndcg_scores = []
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Get unique group IDs
        unique_groups = groups.unique()
        
        for group_id in unique_groups:
            # Get indices for this group
            group_mask = groups == group_id
            group_X = X_scaled[group_mask]
            group_y = y[group_mask]
            
            if len(group_X) == 0 or len(group_y) == 0:
                continue  # Skip empty groups
                
            # Predict scores for this group
            group_scores = self.predict(group_X, return_scores=True)
            
            # Calculate NDCG@k
            if len(group_scores) < at_k:
                # If we have fewer items than k, use the actual length
                actual_k = len(group_scores)
            else:
                actual_k = at_k
                
            try:
                ndcg = ndcg_score([group_y.values], [group_scores], k=actual_k)
                ndcg_scores.append(ndcg)
            except Exception as e:
                print(f"Error calculating NDCG for group {group_id}: {e}")
                continue
        
        if not ndcg_scores:
            return 0.0  # Return 0 if no valid scores
            
        return np.mean(ndcg_scores)

def perform_cross_validation(X, y, groups_for_splitting, df_full_for_group_counts, n_folds=5, cb_params=None):

    print(f"\n--- Performing {n_folds}-Fold Cross-Validation ---")
    gkf = GroupKFold(n_splits=n_folds)
    fold_ndcg_scores = []
    
    default_params = {
        'alpha': 0.2,
        'exploration_strategy': 'ucb',
        'base_algorithm': 'rf',
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 5
    }
    
    current_cb_params = default_params.copy()
    if cb_params:
        current_cb_params.update(cb_params)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups_for_splitting)):
        print(f"--- Fold {fold + 1}/{n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Get group IDs for this fold
        train_groups = groups_for_splitting.iloc[train_idx]
        val_groups = groups_for_splitting.iloc[val_idx]
        
        if len(train_groups.unique()) == 0 or len(val_groups.unique()) == 0:
            print(f"Skipping fold {fold + 1} due to empty train or validation groups/data.")
            continue
        
        # Create and fit the model
        try:
            model = ContextualBanditRanker(**current_cb_params)
            model.fit(X_train, y_train, train_groups)
            
            # Evaluate the model
            ndcg_at_5 = model.evaluate(X_val, y_val, val_groups, at_k=5)
            fold_ndcg_scores.append(ndcg_at_5)
            print(f"Fold {fold + 1} NDCG@5: {ndcg_at_5:.4f}")
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            continue
    
    mean_ndcg = np.mean(fold_ndcg_scores) if fold_ndcg_scores else 0.0
    std_ndcg = np.std(fold_ndcg_scores) if fold_ndcg_scores else 0.0
    
    # Unlike tree-based models, contextual bandits don't have direct feature importances
    # We'll return an empty Series for compatibility
    mean_importances = pd.Series(dtype=float)
    
    print(f"Mean NDCG@5 across {len(fold_ndcg_scores)} folds: {mean_ndcg:.4f} +/- {std_ndcg:.4f}")
    return mean_ndcg, std_ndcg, mean_importances

def tune_hyperparameters_optuna(X, y, groups_for_splitting, df_full_for_group_counts, n_trials=20, n_cv_folds=3):

    print(f"\n--- Tuning Hyperparameters with Optuna ({n_trials} trials, {n_cv_folds} CV folds each) ---")
    
    def objective(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 0.01, 0.5, log=True),
            'exploration_strategy': trial.suggest_categorical('exploration_strategy', ['ucb', 'thompson', 'epsilon_greedy']),
            'base_algorithm': trial.suggest_categorical('base_algorithm', ['rf', 'linear']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10)
        }
        
        # If using random forest, add specific parameters
        if params['base_algorithm'] == 'rf':
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            
        # Using a local variable for clarity within the objective function
        current_X, current_y = X, y 
        current_groups_for_splitting_objective = groups_for_splitting
        
        gkf_optuna = GroupKFold(n_splits=n_cv_folds)
        fold_scores = []
        
        for fold_num, (train_idx, val_idx) in enumerate(gkf_optuna.split(current_X, current_y, groups=current_groups_for_splitting_objective)):
            X_train_trial, X_val_trial = current_X.iloc[train_idx], current_X.iloc[val_idx]
            y_train_trial, y_val_trial = current_y.iloc[train_idx], current_y.iloc[val_idx]
            
            train_groups_trial = current_groups_for_splitting_objective.iloc[train_idx]
            val_groups_trial = current_groups_for_splitting_objective.iloc[val_idx]
            
            if len(train_groups_trial.unique()) == 0 or len(val_groups_trial.unique()) == 0:
                return -1.0 # Penalize this trial heavily
            
            try:
                model_trial = ContextualBanditRanker(**params)
                model_trial.fit(X_train_trial, y_train_trial, train_groups_trial)
                
                # Evaluate the model
                ndcg_at_5 = model_trial.evaluate(X_val_trial, y_val_trial, val_groups_trial, at_k=5)
                fold_scores.append(ndcg_at_5)
            except Exception:
                # If an error occurs during model fitting or evaluation, penalize this trial
                return -1.0
        
        avg_score = np.mean(fold_scores) if fold_scores else 0.0
        return avg_score
    
    study = optuna.create_study(direction='maximize', study_name='contextual_bandit_tuning')
    
    try:
        study.optimize(objective, n_trials=n_trials, timeout=None)
        print(f"Optuna study finished. Best trial NDCG@5: {study.best_value:.4f}")
        print("Best parameters:", study.best_params)
        return study.best_params
    except Exception as e:
        print(f"Error during Optuna optimization: {e}")
        return {}

def train_final_model(X_train_full, y_train_full, groups_train_full, best_params):

    print("\n--- Training Final Model ---")
    
    final_model_params = {
        'alpha': 0.2,
        'exploration_strategy': 'ucb',
        'base_algorithm': 'rf',
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 5
    }
    
    final_model_params.update(best_params)
    
    print("Final model parameters for training:")
    print(final_model_params)
    
    final_model = ContextualBanditRanker(**final_model_params)
    
    # Train the model on the full dataset
    try:
        final_model.fit(X_train_full, y_train_full, groups_train_full)
        print("Final model training completed.")
        return final_model
    except Exception as e:
        print(f"Error during final model training: {e}")
        return None

def predict_and_format_submission(model, X_test, df_test_original_ids, submission_filename="submission_cb.csv"):

    print("\n--- Predicting on Test Data and Formatting Submission ---")
    
    if X_test.empty:
        print("X_test is empty. Cannot make predictions.")
        return
    
    if model is None or not model.is_fitted:
        print("Model is not fitted. Cannot make predictions.")
        return
    
    try:
        # Get predictions (scores) for each hotel
        test_predictions = model.predict(X_test, return_scores=True)
        
        # Add predictions to test DataFrame
        df_test_with_preds = df_test_original_ids.copy()
        
        # Ensure index alignment
        if X_test.index.equals(df_test_with_preds.index):
            df_test_with_preds['predicted_score'] = test_predictions
        else:
            print("Warning: X_test index does not match df_test_original_ids index. Assuming row order is preserved for predictions.")
            if len(X_test) == len(df_test_with_preds):
                df_test_with_preds['predicted_score'] = test_predictions
            else:
                print(f"Error: Length mismatch between X_test ({len(X_test)}) and df_test_original_ids ({len(df_test_with_preds)}). Cannot assign predictions.")
                return
        
        # Format for submission - reorder hotels by predicted scores for each search
        submission_list = []
        
        for srch_id, group_df in df_test_with_preds.groupby('srch_id'):
            ranked_properties = group_df.sort_values('predicted_score', ascending=False)
            for _, row in ranked_properties.iterrows():
                submission_list.append({'SearchId': int(srch_id), 'PropertyId': int(row['prop_id'])})
        
        df_submission = pd.DataFrame(submission_list)
        df_submission.to_csv(submission_filename, index=False)
        
        print(f"Submission file '{submission_filename}' created. Top 5 rows:")
        print(df_submission.head())
    except Exception as e:
        print(f"Error during prediction or submission formatting: {e}")

def save_model(model, filename):

    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):

    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None