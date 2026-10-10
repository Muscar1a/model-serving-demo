import json
from typing import NamedTuple

import joblib
import pandas as pd
from kfp import compiler, dsl
from kfp.dsl import Artifact, Input, Output
from sklearn.metrics import accuracy_score, f1_score


class PrepareDataOutputs(NamedTuple):
    num_records: int
    num_features: int


class TrainModelOutputs(NamedTuple):
    train_accuracy: float
    test_accuracy: float
    eval_accuracy: float
    eval_f1_score: float


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["boto3", "pandas", "pyarrow", "numpy", "scikit-learn"],
)
def prepare_data(
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket_name: str,
    data_prefix: str,
    output_data: Output[Artifact],
) -> PrepareDataOutputs:

    import os
    import tempfile

    import boto3
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    print("=" * 70)
    print("COMPONENT 1: DATA PREPARATION")
    print("=" * 70)

    # Collect from SeaweedFS (S3-compatible)
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        verify=False,
    )

    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=data_prefix)
    parquet_files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    print(f"Found {len(parquet_files)} parquet files")

    dfs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, file_key in enumerate(parquet_files[:10]):
            tmp_file_path = os.path.join(tmp_dir, f"file_{i}.parquet")
            print(f"Downloading file: {file_key} to {tmp_file_path}")
            s3_client.download_file(s3_bucket_name, file_key, tmp_file_path)
            dfs.append(pd.read_parquet(tmp_file_path))

    if not dfs:
        raise Exception("No data found in Seaweedfs!")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data: {combined_df.shape}")

    # Preprocess
    feature_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df_clean = combined_df[feature_cols].copy()
    df_clean = df_clean.fillna(df_clean.median())
    df_clean = df_clean.drop_duplicates()

    # Remove outliers
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

    # Feature engineering
    df_clean["age_x_thalach"] = df_clean["age"] * df_clean["thalach"]
    df_clean["bmi_proxy"] = df_clean["chol"] / (df_clean["thalach"] + 1)

    target = df_clean["target"]
    features = df_clean.drop("target", axis=1)

    # Standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=features.columns)

    df_final = features_df.copy()
    df_final["target"] = target.values

    df_final.to_csv(output_data.path, index=False)

    print(f"[OK] Complete! {len(df_final)} records, {len(features.columns)} features")

    return PrepareDataOutputs(
        num_records=len(df_final), num_features=len(features.columns)
    )


# Component 2: Validate data


@dsl.component(base_image="python:3.9", packages_to_install=["pandas", "numpy"])
def validate_data(
    input_data: Input[Artifact], min_records: int = 100, min_features: int = 10
) -> str:

    print("=" * 70)
    print("COMPONENT2:DATAVALIDATION")
    print("=" * 70)

    df = pd.read_csv(input_data.path)

    validation_checks = {
        "min_records": bool(len(df) >= min_records),
        "min_features": bool((len(df.columns) - 1) >= min_features),
        "no_missing": bool(df.isnull().sum().sum() == 0),
        "balanced_target": bool(df["target"].value_counts().min() >= 30),
    }

    is_valid = all(validation_checks.values())

    validation_report = json.dumps(
        {
            "total_records": int(len(df)),
            "total_features": int(len(df.columns) - 1),
            "checks": validation_checks,
            "is_valid": bool(is_valid),
        },
        indent=2,
    )

    print(f"Totalrecords: {len(df)}")
    print(f"Totalfeatures: {len(df.columns) - 1}")
    print(f"Result: {'[OK]VALID' if is_valid else '[ERROR]INVALID'}")

    if not is_valid:
        raise Exception(f"Validation failed: {validation_report}")

    return validation_report


# COMPONENT 3: Train and Evaluate Model
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"],
)
def train_model(
    input_data: Input[Artifact],
    model_output: Output[Artifact],
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
) -> TrainModelOutputs:

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("COMPONENT3:TRAINING ANDEVALUATION")
    print("=" * 70)

    df = pd.read_csv(input_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Dataset: {X.shape[0]} samples,{X.shape[1]} features")
    print(f"Targetdistribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrainset: {X_train.shape[0]}samples")
    print(f"Testset: {X_test.shape[0]} samples")

    # Hyperparameters
    print("\nHyperparameters:")
    print(f" n_estimators: {n_estimators}")
    print(f" max_depth: {max_depth}")
    print(f" min_samples_split: {min_samples_split}")
    print(f" min_samples_leaf: {min_samples_leaf}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTraining model...")
    import time

    start_time = time.time()

    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"[OK] Training completed in {elapsed_time:.2f} seconds")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    eval_acc = accuracy_score(y_test, test_pred)
    eval_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    print("\nTrainingResults:")
    print(f" TrainAccuracy: {train_acc:.4f}")
    print(f" TestAccuracy: {test_acc:.4f}")

    print("\nEvaluationResults:")
    print(f" Accuracy: {eval_acc:.4f}")
    print(f" F1-Score: {eval_f1:.4f}")

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 5 Importance Features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    joblib.dump(model, model_output.path)
    print(f"\n[OK] Model saved to: {model_output.path}")

    return TrainModelOutputs(
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        eval_accuracy=float(eval_acc),
        eval_f1_score=float(eval_f1),
    )


# Component 4: RTegister Model to Seaweedfs
@dsl.component(base_image="python:3.9", packages_to_install=["boto3", "joblib"])
def register_model(
    model: Input[Artifact],
    train_accuracy: float,
    test_accuracy: float,
    eval_accuracy: float,
    eval_f1_score: float,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket_name: str,
    n_estimators: int,
    max_depth: int,
) -> str:
    from datetime import datetime

    import boto3

    print("=" * 70)
    print("COMPONENT5:MODELREGISTRY")
    print("=" * 70)

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        verify=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"v_{timestamp}_acc{eval_accuracy:.4f}".replace(".", "_")

    print(f"Model version: {model_version}")

    model_key = f"models/heart-disease/{model_version}/model.joblib"
    with open(model.path, "rb") as f:
        s3_client.upload_fileobj(f, s3_bucket_name, model_key)
    print(f"[OK] Model uploaded:  s3://{s3_bucket_name}/{model_key}")

    metadata = {
        "model_version": model_version,
        "timestamp": timestamp,
        "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth},
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "eval_accuracy": float(eval_accuracy),
            "eval_f1_score": float(eval_f1_score),
        },
        "model_type": "RandomForestClassifier",
        "framework": "scikit-learn",
        "dataset": "heart-disease",
    }

    metadata_key = f"models/heart-disease/{model_version}/metadata.json"
    s3_client.put_object(
        Bucket=s3_bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode(),
    )
    print(f"[OK] Metadata uploaded: s3://{s3_bucket_name}/{metadata_key}")

    model_uri = f"s3://{s3_bucket_name}/{model_key}"

    print("\n[OK] Model registration complete!")
    print(f"Model URI: {model_uri}")
    print(f"Test Accuracy: {eval_accuracy:.4f}")
    print(f"F1-Score: {eval_f1_score:.4f}")

    return model_uri


# ============================================================================
# PIPELINE
# ============================================================================
@dsl.pipeline(
    name="heart-disease-simple-no-tuning",
    description="Simple pipeline: Prepare -> Validate -> Train and Evaluate -> Register",
)
def heart_disease_pipeline(
    # Seaweedfs configs
    s3_endpoint: str = "http://192.168.2.4:19000",
    s3_access_key: str = "seaweedadmin",
    s3_secret_key: str = "seaweedadmin",
    s3_bucket_name: str = "ml-data",
    data_prefix: str = "raw/heart-disease",
    # Model hyperparameters (FIXED - no tuning)
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
):

    # ========== STEP 1: DATA PREPARATION ==========
    prepare_task = prepare_data(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket_name=s3_bucket_name,
        data_prefix=data_prefix,
    )
    prepare_task.set_display_name("Data Preparation")

    # ========== STEP 2: DATA VALIDATION ==========
    validate_task = validate_data(input_data=prepare_task.outputs["output_data"])
    validate_task.set_display_name("Data Validation")
    validate_task.after(prepare_task)

    # ========== STEP 3: TRAINING AND EVALUATION ==========
    train_task = train_model(
        input_data=prepare_task.outputs["output_data"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    train_task.set_display_name("Training and Evaluation")
    train_task.after(validate_task)

    # ========== STEP 4: MODEL REGISTRY ==========
    register_task = register_model(
        model=train_task.outputs["model_output"],
        train_accuracy=train_task.outputs["train_accuracy"],
        test_accuracy=train_task.outputs["test_accuracy"],
        eval_accuracy=train_task.outputs["eval_accuracy"],
        eval_f1_score=train_task.outputs["eval_f1_score"],
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket_name=s3_bucket_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    register_task.set_display_name("Model Registry")
    register_task.after(train_task)


# ============================================================================
# COMPILE PIPELINE
# ============================================================================
if __name__ == "__main__":
    output_file = "heart_disease_pipeline_simple.yaml"

    print("=" * 70)
    print("COMPILING SIMPLE PIPELINE (NO HYPERPARAMETER TUNING)")
    print("=" * 70)

    compiler.Compiler().compile(
        pipeline_func=heart_disease_pipeline, package_path=output_file
    )

    print("\n[OK] Pipeline compiled successfully!")
    print(f"Output file: {output_file}")

    print("\nPipeline Flow:")
    print("  1. Data Preparation (SeaweedFS S3 -> CSV)")
    print("  2. Data Validation")
    print("  3. Training and Evaluation (RandomForest with FIXED hyperparameters)")
    print("  4. Model Registry (save to SeaweedFS S3)")

    print("\nDefault Hyperparameters:")
    print("  - n_estimators: 100")
    print("  - max_depth: 10")
    print("  - min_samples_split: 5")
    print("  - min_samples_leaf: 2")

    print("\nTo run with custom hyperparameters:")
    print("  Upload to Kubeflow UI and modify parameters in the run")

    print("=" * 70)
