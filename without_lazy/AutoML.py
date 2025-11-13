import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from context_tree_selector import ContextTreeFeatureSelector
from context_tree_selector import LLMFeatureSelector # Make sure to import the new class
from feature_selection_ga import GeneticFeatureSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
from hardcoded_model_selection import find_best_model

# def run_automl_pipeline(df: pd.DataFrame, target_column: str, mode: str = 'full') -> dict:
def run_automl_pipeline(df: pd.DataFrame, target_column: str, llm_client, mode: str = 'full') -> dict:
    """
    Runs an AutoML experiment with a specified feature selection strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable.
        mode (str): The feature selection mode. One of:
                    'full' -> Context Tree + Genetic Algorithm.
                    'ga_only' -> Genetic Algorithm only.
                    'direct' -> No feature selection.

    Returns:
        dict: A dictionary with all results for reporting.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(" AutoML step skipped: The provided DataFrame is empty or invalid.")
        return {}

    print(f"\n\n===== STARTING AutoML RUN: MODE = '{mode.upper()}' =====")

    df_processed = df.copy()
    df_processed.dropna(subset=[target_column], inplace=True)
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- Scaling Numerical Features ---
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        print(f"\nApplied StandardScaler to {len(numeric_cols)} columns.")

    # --- Determine Problem Type ---
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        problem_type = "Regression"
        ga_estimator = LinearRegression()
        ga_scoring = 'r2'
    else:
        problem_type = "Classification"
        ga_estimator = LogisticRegression(random_state=123, max_iter=1000)
        ga_scoring = 'accuracy'
    print(f"Problem type detected as: {problem_type}.")

    # --- Feature Selection Pipeline ---
    context_selected_features = []
    final_selected_features = []
    X_train_final, X_test_final = X_train.copy(), X_test.copy()

    if mode == 'Genetic':
        print("\n--- Running PHASE 1: Context Tree Feature Selection ---")
        # the below is the context_tree Method Call(): (now, I changed it to the context awareness); 
        # context_tree_selector = ContextTreeFeatureSelector(X=X_train, y=y_train, problem_type=problem_type.lower(), random_state=123)
        context_tree_selector = LLMFeatureSelector(X=X_train, y=y_train, problem_type=problem_type, target_column=target_column, llm_client=llm_client)
        context_selected_features, _ = context_tree_selector.run()
        # X_train_context = X_train[context_selected_features]
        # X_test_context = X_test[context_selected_features]
        
        # print("\n--- Running PHASE 2: Genetic Algorithm Feature Selection ---")
        # ga_selector = GeneticFeatureSelector(model=ga_estimator, X=X_train_context, y=y_train, scoring=ga_scoring, cv=3, random_state=123)
        # final_selected_features = ga_selector.run()
        final_selected_features = context_selected_features; 
        X_train_final = X_train[final_selected_features]
        X_test_final = X_test[final_selected_features]

    elif mode == 'Genetic + Context Aware':
        print("\n--- Skipping Context Tree ---")
        print("\n--- Running PHASE 2: Genetic Algorithm Feature Selection (on all features) ---")
        ga_selector = GeneticFeatureSelector(model=ga_estimator, X=X_train, y=y_train, scoring=ga_scoring, cv=3, random_state=123)
        final_selected_features = ga_selector.run()
        
        X_train_final = X_train[final_selected_features]
        X_test_final = X_test[final_selected_features]

    elif mode == 'Genetic + Context Tree':
        print("\n--- Skipping All Feature Selection ---")
        final_selected_features = X_train.columns.tolist()

    print(f"\nNumber of features used for modeling: {len(final_selected_features)}")

    numeric_features = X_train_final.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    X_train_final = X_train_final[numeric_features]
    X_test_final = X_test_final[numeric_features]

    # --- Model Selection and Training ---
    print(f"\n--- Running PHASE 3: Model Selection and Training ---")
    _, best_model_name, report_metrics = find_best_model(X_train_final, y_train, X_test_final, y_test, problem_type)

    print(f"\n===== AutoML RUN COMPLETE: MODE = '{mode.upper()}' =====")
    
    if mode == "Genetic":
        report_data = {
            "mode": mode,
            "context_tree_features": context_selected_features,
            "genetic_algorithm_features": final_selected_features,
            "best_model_name": best_model_name,
            "model_metrics": report_metrics,
            "problem_type": problem_type
        }
    elif mode == "Genetic + Context Aware":
        report_data = {
            "mode": mode,
            "context_tree_features": context_selected_features,
            "genetic_algorithm_features": final_selected_features,
            "best_model_name": best_model_name,
            "model_metrics": report_metrics,
            "problem_type": problem_type
        }
    elif mode == "Genetic + Context Tree": 
        report_data = {
            "mode": mode,
            "context_tree_features": context_selected_features,
            "genetic_algorithm_features": final_selected_features,
            "best_model_name": best_model_name,
            "model_metrics": report_metrics,
            "problem_type": problem_type
        }
    return report_data
