import sys
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(tracking_uri)


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ingest import load_data
    from src.preprocess import clean_data, balance_data
    from src.features import apply_feature_cross, apply_hashing
except ImportError:
    sys.exit(1)

warnings.filterwarnings("ignore")

class MLEngineerPipeline:
    def __init__(self, processed_dataframe, experiment_name="Course_Completion_MLOps"):
        self.data = processed_dataframe
        self.experiment_name = experiment_name
        self.results = []

        mlflow.set_experiment(self.experiment_name)

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

    def run_classification_experiments(self):
        target_col = 'target' if 'target' in self.data.columns else 'Completed'

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        drop_cols = [target_col, 'Progress_Percentage']
        cols_to_drop = [c for c in drop_cols if c in self.data.columns]

        X = self.data.drop(columns=cols_to_drop)
        y = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "RandomForest_Bagging": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
            "XGBoost_Boosting": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42)
        }

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds)

                mlflow.log_param("model_type", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)

                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="CourseCompletionModel")
    

                self.results.append({
                    "Model": name,
                    "Task": "Classification",
                    "Accuracy": acc,
                    "F1": f1,
                    "RMSE": None
                })
                print(f"  {name} -> Accuracy: {acc:.4f}")

    def run_reframing_experiment(self):
        if 'Progress_Percentage' not in self.data.columns:
            return

        target_col = 'target' if 'target' in self.data.columns else 'Completed'

        cols_to_drop = [target_col, 'Progress_Percentage']
        real_drop = [c for c in cols_to_drop if c in self.data.columns]

        X = self.data.drop(columns=real_drop)
        y_reg = self.data['Progress_Percentage']
        y_class_true = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, _, y_test_class = train_test_split(X, y_class_true, test_size=0.2, random_state=42)

        model_name = "XGBoost_Reframed_Regressor"
        model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            preds_percent = model.predict(X_test)

            preds_class = [1 if p >= 50.0 else 0 for p in preds_percent]

            rmse = np.sqrt(mean_squared_error(y_test, preds_percent))
            acc = accuracy_score(y_test_class, preds_class)

            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("derived_accuracy", acc)
            
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="CourseCompletionModel_Reframed")
    

            self.results.append({
                "Model": model_name,
                "Task": "Reframing (Reg->Clf)",
                "Accuracy": acc,
                "F1": None,
                "RMSE": rmse
            })
            print(f"  {model_name} -> Derived Accuracy: {acc:.4f}")

    def get_results_table(self):
        return pd.DataFrame(self.results).sort_values(by="Accuracy", ascending=False)

if __name__ == "__main__":
    DATA_PATH = "data/Course_Completion_Prediction.csv"

    try:
        raw_df = load_data(DATA_PATH)
        clean_df = clean_data(raw_df)
        balanced_df = balance_data(clean_df)
        df_crossed = apply_feature_cross(balanced_df)

        if 'Student_ID' in df_crossed.columns:
            final_df = apply_hashing(df_crossed, 'Student_ID', n_features=50)
        else:
            final_df = df_crossed

        object_cols = final_df.select_dtypes(include=['object']).columns
        le = LabelEncoder()

        for col in object_cols:
            if col != 'target' and 'hashed' not in col:
                final_df[col] = le.fit_transform(final_df[col].astype(str))

        print(f"Data Ready. Shape: {final_df.shape}")

        pipeline = MLEngineerPipeline(final_df)
        pipeline.run_classification_experiments()
        pipeline.run_reframing_experiment()

        if len(pipeline.results) == 0:
            raise RuntimeError("No models were trained successfully")
