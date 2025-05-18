# Expedia Hotel Recommendations

This repository contains code for the Expedia Hotel Recommendations Kaggle competition, developed as part of the Data Mining Techniques course at Vrije Universiteit Amsterdam.

## Project Structure

- `data_understanding.ipynb`: Exploratory data analysis on the Expedia dataset
- `data_imputation.ipynb`: Handling missing values in the dataset
- `data_feature_engineering.ipynb`: Feature engineering for improving model performance
- `main.ipynb`: Main workflow using LambdaMART (Learning to Rank) with LightGBM
- `reinforcement.ipynb`: Alternative approach using Contextual Bandits (Reinforcement Learning)

## Featured Approaches

### 1. LambdaMART Ranking Model
The original approach uses LightGBM for learning to rank:
- Optimizes NDCG@5 metric for ranking quality
- Performs cross-validation using GroupKFold to respect search groups
- Uses hyperparameter tuning with Optuna

### 2. Contextual Bandits Model
An experimental approach using reinforcement learning:
- Models hotel recommendation as an exploration-exploitation problem
- Balances between showing known good hotels and exploring new options
- Provides a framework for online learning

## Feature Engineering
The `data_feature_engineering.ipynb` notebook creates a rich set of features:

1. **Time-based Features**: Extracts temporal patterns from timestamps
2. **User-based Features**: Captures user preferences and behaviors
3. **Property-based Features**: Enhances hotel characteristics representation
4. **Price/Value Features**: Creates insightful pricing metrics
5. **Competitive Position Features**: Shows how hotels compare to competitors
6. **Interaction Features**: Captures relationships between multiple attributes

## How to Run

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

### Execution Order
1. Run `data_understanding.ipynb` for exploratory analysis
2. Run `data_imputation.ipynb` to handle missing values
3. Run `data_feature_engineering.ipynb` to create enhanced features
4. Run `main.ipynb` for the LambdaMART approach
5. (Optional) Run `reinforcement.ipynb` for the contextual bandits approach

### Data Files
Place the Expedia dataset files in a `data` directory:
- training_set_VU_DM.csv
- test_set_VU_DM.csv

## Results

The feature-engineered model achieves significantly improved NDCG@5 scores compared to the baseline model. Key findings include:

- Property location scores are the most influential features
- Price-related features like price normalization within search are highly important
- Temporal patterns around booking windows and seasonality provide helpful signals
- User-property interaction features improve model performance

## Future Work

- Implement ensemble methods combining ranking and contextual bandits approaches
- Add more complex feature interactions using neural networks
- Explore session-based recommendation techniques