import pandas as pd
import json
import re
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LLMFeatureSelector:
    """
    Selects features using an LLM that analyzes the data's context.

    This enhanced selector provides the LLM with column statistics,
    pre-calculated feature importance scores, and correlation data
    to make a highly informed, context-aware feature selection.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, problem_type: str, target_column: str, llm_client, max_features=100, random_state=42):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.target_column = target_column
        self.llm_client = llm_client
        self.max_features = max_features
        self.random_state = random_state

    def _get_column_statistics(self):
        """Generates a summary of statistics for each column."""
        stats = []
        for col in self.X.columns:
            dtype = str(self.X[col].dtype)
            unique_vals = self.X[col].nunique()
            missing_pct = self.X[col].isnull().sum() / len(self.X) * 100
            stats.append(f"- `{col}` (Type: {dtype}, Unique Values: {unique_vals}, Missing: {missing_pct:.1f}%)")
        return "\n".join(stats)

    # NEW: Method to calculate preliminary feature importance
    def _get_feature_importance(self):
        """Calculates feature importance using a RandomForest model."""
        print("  - Calculating preliminary feature importance...")
        X_encoded = self.X.copy()
        # Encode categorical features for the model
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

        if self.problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        
        model.fit(X_encoded, self.y)
        importances = pd.Series(model.feature_importances_, index=self.X.columns)
        return importances.sort_values(ascending=False)

    # NEW: Method to find highly correlated features
    def _get_correlation_info(self, threshold=0.9):
        """Finds and summarizes pairs of highly correlated features."""
        print("  - Calculating feature correlations...")


        numeric_X = self.X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        corr_matrix = numeric_X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col, upper_tri[col].idxmax(), upper_tri[col].max()) for col in upper_tri.columns if upper_tri[col].max() > threshold]
        
        if not high_corr_pairs:
            return "No feature pairs have a correlation coefficient above 0.9."
            
        summary = "Highly Correlated Feature Pairs (potential redundancy):\n"
        for col1, col2, corr_val in high_corr_pairs:
            summary += f"- `{col1}` and `{col2}` (Correlation: {corr_val:.2f})\n"
        return summary

    def _create_prompt(self):
        """Creates the detailed prompt for the LLM."""
        column_stats = self._get_column_statistics()
        data_sample = pd.concat([self.X, self.y], axis=1).head(3).to_string()
        
        # NEW: Get importance and correlation data
        feature_importances = self._get_feature_importance()
        correlation_info = self._get_correlation_info()

        # NEW: Format importance scores for the prompt
        importance_summary = "Preliminary Feature Importance (from RandomForest model):\n"
        for feature, score in feature_importances.head(15).items(): # Show top 15
            importance_summary += f"- `{feature}`: {score:.4f}\n"

        # NEW: Enhanced prompt with Chain-of-Thought and richer context
        prompt = f"""
        You are an expert data scientist specializing in feature engineering for high-performance machine learning models.
        Your task is to select the most predictive, relevant, and non-redundant features from a dataset.

        **1. Task Context:**
        - **Target Variable (what to predict):** `{self.target_column}`
        - **Problem Type:** `{self.problem_type}`
        - **Maximum Number of Features to Select:** `{self.max_features}`

        **2. Available Feature Statistics:**
        {column_stats}

        **3. Preliminary Feature Importance Analysis:**
        The following features were ranked by a RandomForest model. This is a strong hint, but your contextual understanding is needed for the final selection.
        {importance_summary}

        **4. Redundancy Analysis (Correlation):**
        The following feature pairs are highly correlated. You should likely select only one feature from each pair.
        {correlation_info}

        **5. Data Sample:**
        Here is a small sample of the data:
        {data_sample}

        **Your Instructions (Please follow carefully):**

        **Step 1: Reasoning (Chain of Thought):**
        First, write down your analysis in a <reasoning> block. Analyze the provided information to build your selection strategy. Consider the following:
        - Which features have the highest preliminary importance?
        - Are any top features redundant (highly correlated)? If so, which one from the pair is better to keep? (Consider its type, missing values, and interpretability).
        - Are there any features with low importance that might still be valuable due to unique contextual information not captured by the model?
        - Does your final selection align with the goal of predicting `{self.target_column}`?

        **Step 2: Final Selection (JSON Output):**
        After your reasoning, provide your final selection as a single, valid JSON list of strings. The list should contain the feature names you've chosen.
        For example: ["feature1", "feature2", "feature3"]
        Provide ONLY the JSON list in a <json> block. Do not add any other text after this block.
        """
        return prompt

    def run(self):
        """Runs the LLM-based feature selection process."""
        print("\n--- Running Enhanced LLM-Powered Feature Selection ---")
        prompt = self._create_prompt()

        try:
            print("  - Sending request to LLM for feature analysis...")
            response_text = self.llm_client.invoke(prompt).content
            
            # NEW: Updated parsing logic for Chain-of-Thought response
            print("  - Parsing LLM response...")
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
            json_match = re.search(r'<json>(.*?)</json>', response_text, re.DOTALL)

            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                print("  - LLM Reasoning:\n", reasoning)
            else:
                print("  - Warning: Could not find <reasoning> block in the response.")

            if not json_match:
                st.warning("LLM did not return a valid JSON block. Using all features as a fallback.")
                print("  - Error: Could not find a `<json>[...]</json>` block in the LLM response.")
                return self.X.columns.tolist(), {}

            json_string = json_match.group(1).strip()
            selected_features = json.loads(json_string)

            if not isinstance(selected_features, list) or not all(isinstance(i, str) for i in selected_features):
                st.warning("LLM response was not a valid list of feature names. Using all features as a fallback.")
                print(f"  - Error: LLM output was not a list of strings. Type: {type(selected_features)}")
                return self.X.columns.tolist(), {}

            valid_features = [f for f in selected_features if f in self.X.columns]
            print(f"  - LLM selected {len(valid_features)} valid features.")
            
            return valid_features, {}

        except (json.JSONDecodeError, Exception) as e:
            st.error(f"An error occurred during LLM feature selection: {e}")
            print(f"  - Error during LLM feature selection: {e}")
            return self.X.columns.tolist(), {}
