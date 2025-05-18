"""
Main Script for Expedia Hotel Booking Prediction

Assignment 2: Data Mining Techniques, Vrije Universiteit Amsterdam
"""

import os
import sys
import warnings

# Try to import required libraries 
try:
    import pandas as pd
    import numpy as np
    print("Successfully imported pandas and numpy")
except ImportError as e:
    print(f"ERROR: Missing basic dependency: {e}")
    print("Please install required packages using: pip install pandas numpy")
    sys.exit(1)

# Set default flag values
IMPORTS_SUCCESSFUL = False
LIGHTGBM_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.model_selection import GroupKFold
    print("Successfully imported scikit-learn")
except ImportError as e:
    print(f"ERROR: Missing scikit-learn: {e}")
    print("Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

# Try importing lightgbm with a timeout workaround
print("Attempting to import lightgbm (this may take a moment)...")
try:
    # Need to set a timeout for the import because it can hang
    import importlib.util
    import sys
    import threading
    import time
    
    # Define a function to attempt the import with a timeout
    def import_with_timeout(module_name, timeout=10):
        result = {"module": None, "success": False, "error": None}
        
        def _import():
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    result["error"] = f"Module {module_name} not found"
                    return
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                result["module"] = module
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
        
        # Run the import in a thread
        thread = threading.Thread(target=_import)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(timeout)
        
        if thread.is_alive():
            return False, f"Import timed out after {timeout} seconds"
        
        if result["success"]:
            return True, result["module"]
        else:
            return False, result["error"]
    
    # Try to import lightgbm with a timeout
    success, lgb_result = import_with_timeout("lightgbm", timeout=5)
    
    if success:
        lgb = lgb_result
        print("Successfully imported lightgbm")
        LIGHTGBM_AVAILABLE = True
        
        # Now try to import our model functions
        try:
            import lightgbm_ranker_model as lgbm_model
            print("Successfully imported custom model module")
            IMPORTS_SUCCESSFUL = True
        except ImportError as e:
            print(f"ERROR: Failed to import lightgbm_ranker_model: {e}")
            print("Please ensure lightgbm_ranker_model.py is in the current directory")
    else:
        print(f"ERROR: Lightgbm import timed out or failed: {lgb_result}")
        print("Will try to proceed without LightGBM. You might need to install it: pip install lightgbm")
except Exception as e:
    print(f"ERROR: Unexpected problem with imports: {e}")
    print("Will try to proceed with limited functionality")

# Suppress warnings from Optuna/LightGBM
warnings.filterwarnings('ignore', message='Found \'eval_at\' in params.*')

# --- Configuration ---
DATA_DIR = '../data.nosync'
# Point to our feature-engineered dataset
TRAIN_FILE = 'imputed_sample_data_with_features.csv'
# Check if feature-engineered dataset exists and use it if available
if os.path.exists(TRAIN_FILE):
    print(f"Using feature-engineered dataset: {TRAIN_FILE}")
else:
    # Fallback to original file
    TRAIN_FILE = os.path.join(DATA_DIR, 'train_imputed.csv')
    print(f"Feature-engineered dataset not found. Using: {TRAIN_FILE}")

TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_FILENAME = 'submission_feature_engin.csv'

SAMPLE_FRACTION = 0.1
N_FOLDS_CV = 5
N_FOLDS_TUNING = 3
N_OPTUNA_TRIALS = 20
RANDOM_STATE = 42

def main():
    # Check for successful imports
    if not LIGHTGBM_AVAILABLE:
        print("\nWARNING: LightGBM not available - cannot run model training and evaluation.")
        print("We can still preprocess the data, but model training will be skipped.")
        print("Please install LightGBM to enable full functionality.")
        print("You can try: pip install lightgbm")
        
        # Skip interactive input in non-interactive environments
        if not sys.stdout.isatty():
            print("Running in non-interactive mode. Continuing with limited functionality.")
        else:
            # Ask user if they want to continue with limited functionality
            proceed = input("\nDo you want to continue with data preprocessing only? (y/n): ")
            if proceed.lower() != 'y':
                print("Exiting as requested.")
                return
        
    # --- 1. Load Data ---
    print("Loading training data...")
    try:
        df_train_full = pd.read_csv(TRAIN_FILE)
        print(f"Loaded training data with shape: {df_train_full.shape}")
    except FileNotFoundError:
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # --- 2. Create Relevance Score if not already present ---
    if 'relevance' not in df_train_full.columns:
        print("\nCreating relevance score...")
        df_train_full['relevance'] = 0
        df_train_full.loc[df_train_full['click_bool'] == 1, 'relevance'] = 1
        df_train_full.loc[df_train_full['booking_bool'] == 1, 'relevance'] = 2
    
    print("Relevance score distribution:")
    print(df_train_full['relevance'].value_counts())

    # --- 3. Data Sampling (Group-aware) ---
    print(f"\nSampling {SAMPLE_FRACTION*100}% of the data based on srch_id...")
    unique_srch_ids = df_train_full['srch_id'].unique()
    sampled_srch_ids_count = int(len(unique_srch_ids) * SAMPLE_FRACTION)
    sampled_srch_ids = np.random.choice(unique_srch_ids, size=sampled_srch_ids_count, replace=False)
    df_sample = df_train_full[df_train_full['srch_id'].isin(sampled_srch_ids)].copy()
    print(f"Sampled data shape: {df_sample.shape}")
    del df_train_full  # Free up memory

    # Initialize feature variables
    X = None
    y = None
    groups_for_splitting = None
    feature_columns = []

    # --- 4. Feature Selection & Preparation ---
    print("\nDefining feature set and preparing X, y, groups...")
    
    # Basic features from original dataset
    basic_features = [
        'visitor_location_country_id', 'prop_country_id',
        'prop_starrating', 'prop_review_score', 'prop_brand_bool',
        'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
        'price_usd', 'promotion_flag', 'orig_destination_distance'
    ]
    
    # Time-based features from feature engineering
    time_features = [
        'year', 'month', 'day', 'dayofweek', 'hour', 'season', 'is_weekend', 
        'day_part', 'days_since_first_date', 'checkin_month', 'checkin_dayofweek', 
        'checkin_season', 'is_holiday_season'
    ]
    
    # User-based features from feature engineering
    user_features = [
        'has_user_history', 'star_rating_match', 'price_match', 'is_domestic_search',
        'total_travelers', 'has_children', 'is_family', 'rooms_per_person',
        'is_short_stay', 'is_long_stay', 'is_last_minute', 'is_early_booking',
        'price_vs_country_avg', 'stay_vs_country_avg'
    ]
    
    # Property-based features from feature engineering
    property_features = [
        'hotel_quality', 'star_review_gap', 'combined_location_score', 
        'location_quality', 'distance_category', 'prop_popularity', 
        'prop_country_rank', 'prop_country_rank_pct', 'prop_matches_destination'
    ]
    
    # Price and value features from feature engineering
    price_features = [
        'price_per_person', 'price_per_night', 'price_per_room',
        'value_for_money', 'value_for_money_normalized', 'price_normalized',
        'price_rank', 'price_rank_pct', 'price_tier', 'price_zscore',
        'has_promotion', 'log_price_ratio', 'price_discount'
    ]
    
    # Competitive position features from feature engineering
    comp_features = [
        'comp_rate_available', 'comp_inv_available', 'competitors_count',
        'better_price_count', 'worse_price_count', 'price_comp_ratio',
        'avg_price_diff', 'comp_advantage'
    ]
    
    # Interaction features from feature engineering
    interaction_features = [
        'user_prop_country_match', 'star_rating_for_price', 'review_for_price',
        'location_for_price', 'family_friendly_score', 'business_travel_score',
        'vacation_score', 'quality_price_ratio'
    ]
    
    # PCA components (if available)
    pca_features = [f'pca_component_{i+1}' for i in range(20)]
    
    # Combine all feature groups
    all_feature_groups = [
        ("Basic", basic_features),
        ("Time", time_features),
        ("User", user_features),
        ("Property", property_features),
        ("Price", price_features),
        ("Competitive", comp_features),
        ("Interaction", interaction_features),
        ("PCA", pca_features)
    ]
    
    # Find which features are actually available in our dataset
    feature_columns = []
    for group_name, group_features in all_feature_groups:
        available_features = [col for col in group_features if col in df_sample.columns]
        feature_columns.extend(available_features)
        if available_features:
            print(f"✓ Added {len(available_features)}/{len(group_features)} {group_name} features")
        else:
            print(f"✗ No {group_name} features found in dataset")
            
    # Remove duplicate features if any
    feature_columns = list(dict.fromkeys(feature_columns))
    
    # If no engineered features are available, fall back to basic features only
    if not feature_columns:
        print("Warning: No feature columns found. Falling back to basic features...")
        # Ensure basic features exist in the dataset
        fallback_features = [col for col in basic_features if col in df_sample.columns]
        if not fallback_features:
            print("Error: No basic features available in dataset. Stopping.")
            return
        feature_columns = fallback_features
        print(f"Using {len(feature_columns)} basic features: {feature_columns}")
    
    X = df_sample[feature_columns].copy()
    y = df_sample['relevance'].copy()
    groups_for_splitting = df_sample['srch_id']  # Used by GroupKFold for splitting

    # Basic Imputation for any remaining missing values
    print("Performing basic median imputation for numerical features in X...")
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
    
    print(f"Selected {len(feature_columns)} features total")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"NaNs remaining in X after imputation: {X.isnull().sum().sum()}")
    print(f"Number of unique groups for splitting: {groups_for_splitting.nunique()}")

    # --- 5. Model Training and Evaluation (if LightGBM is available) ---
    final_model = None
    
    if not LIGHTGBM_AVAILABLE:
        print("\nSkipping model training and evaluation as LightGBM is not available.")
        print("The script will save the preprocessed data for future use.")
        
        # Save the preprocessed data for future use
        processed_data_path = "preprocessed_expedia_data.csv"
        print(f"\nSaving preprocessed data to {processed_data_path}...")
        
        # Add all engineered features to the original data
        df_sample_with_features = df_sample.copy()
        df_sample_with_features = pd.concat([df_sample_with_features, X], axis=1)
        
        # Save to CSV
        df_sample_with_features.to_csv(processed_data_path, index=False)
        print(f"Preprocessed data saved successfully with {len(df_sample_with_features.columns)} columns.")
        print("You can use this data later once LightGBM is installed.")
        
        # Exit early since we can't continue with model training
        return
    else:
        # Continue with model training if LightGBM is available
        initial_lgbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE
        }
        
        # --- 5a. Cross-Validation ---
        print(f"\n--- Performing {N_FOLDS_CV}-Fold Cross-Validation ---")
        mean_cv_ndcg, std_cv_ndcg, cv_feature_importances = lgbm_model.perform_cross_validation(
            X, y, 
            groups_for_splitting=groups_for_splitting,
            df_full_for_group_counts=df_sample,
            n_folds=N_FOLDS_CV,
            lgbm_params=initial_lgbm_params
        )
        
        print(f"\nCross-Validation Mean NDCG@5: {mean_cv_ndcg:.4f} +/- {std_cv_ndcg:.4f}")
        print("\nTop 20 Feature Importances from CV:")
        print(cv_feature_importances.head(20))
    
        # --- 5b. Hyperparameter Tuning ---
        print(f"\n--- Tuning Hyperparameters with Optuna ({N_OPTUNA_TRIALS} trials) ---")
        best_params = lgbm_model.tune_hyperparameters_optuna(
            X, y, 
            groups_for_splitting=groups_for_splitting,
            df_full_for_group_counts=df_sample,
            n_trials=N_OPTUNA_TRIALS, 
            n_cv_folds=N_FOLDS_TUNING
        )
        
        if best_params:
            print("\nBest parameters found by Optuna:")
            for key, value in best_params.items():
                print(f"    {key}: {value}")
        else:
            print("\nOptuna tuning did not return parameters. Using default parameters.")
            best_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 7,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
    
        # --- 5c. Train Final Model ---
        print("\n--- Training Final Model with Best Parameters ---")
        groups_train_full = df_sample.groupby('srch_id').size().to_numpy()
        
        final_model = lgbm_model.train_final_model(
            X_train_full=X,
            y_train_full=y,
            groups_train_full=groups_train_full,
            df_full_for_group_counts=df_sample,
            best_params=best_params
        )
        
        if final_model:
            print("Final model successfully trained.")
        else:
            print("Final model training failed.")
            return

    # --- 6. Prepare Test Data and Generate Submission (if model is available) ---
    if final_model is None:
        print("\nSkipping test prediction and submission as no model is available.")
        return
        
    print("\n--- Preparing Test Data and Generating Submission ---")
    
    # Load Test Data
    try:
        df_test = pd.read_csv(TEST_FILE)
        print(f"Loaded test dataset with shape: {df_test.shape}")
    except FileNotFoundError:
        print(f"ERROR: Test file not found at {TEST_FILE}")
        return
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Preprocess Test Data
    print("Preprocessing test data...")
    # Check which feature columns exist in test data
    missing_features = [col for col in feature_columns if col not in df_test.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in test data: {missing_features[:5]}...")
        print("Creating these features with default values.")
        
    # Create missing columns with NaN values
    for col in feature_columns:
        if col not in df_test.columns:
            df_test[col] = np.nan
    
    # Create X_test with only the selected features
    X_test = df_test[feature_columns].copy()
    
    # Impute missing values using medians from training data
    print("Imputing missing values in test data...")
    nan_counts_before = X_test.isnull().sum().sum()
    if nan_counts_before > 0:
        print(f"Found {nan_counts_before} missing values to impute")
        
    for col in X_test.columns:
        if X_test[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_test[col]):
                train_median = X[col].median()
                X_test[col] = X_test[col].fillna(train_median)
                print(f"  Imputed feature '{col}' with median: {train_median}")
            else:
                # For non-numeric columns, use mode imputation
                train_mode = X[col].mode().iloc[0] if not X[col].mode().empty else 0
                X_test[col] = X_test[col].fillna(train_mode)
                print(f"  Imputed non-numeric feature '{col}' with mode: {train_mode}")
    
    nan_counts_after = X_test.isnull().sum().sum()
    print(f"NaNs remaining in X_test after imputation: {nan_counts_after}")
    
    if nan_counts_after > 0:
        print("Warning: Some NaN values remain. Using additional fallback imputation...")
        X_test = X_test.fillna(0)  # Final fallback to ensure no NaNs remain
    
    # Generate submission file
    lgbm_model.predict_and_format_submission(
        model=final_model,
        X_test=X_test,
        df_test_original_ids=df_test,
        submission_filename=SUBMISSION_FILENAME
    )
    
    print(f"Completed! Submission file saved as {SUBMISSION_FILENAME}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running main script: {e}")
        print("Check that all dependencies are installed and data paths are correct.")
        import traceback
        traceback.print_exc()