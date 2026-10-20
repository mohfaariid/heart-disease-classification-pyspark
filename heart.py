from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def main():
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("HeartDiseaseClassification_PySpark") \
        .getOrCreate()
    
    print("=== Heart Disease Classification with PySpark ===")

    # 2. Load dataset
    data_path = "heart.csv"
    data = spark.read.csv(data_path, header=True, inferSchema=True)
    print(f"Total Rows: {data.count()}, Total Features: {len(data.columns) - 1}")

    # 3. Check missing values
    missing_counts = data.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns
    ])
    print("Missing value scan completed.")

    # 4. Feature Vectorization & Scaling
    feature_cols = [c for c in data.columns if c != 'target']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

    # 5. Split train/test (80/20)
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Train set: {train_data.count()}, Test set: {test_data.count()}")

    # 6. Benchmark Models
    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="target", maxIter=20),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="target", numTrees=50, seed=42),
        "Gradient-Boosted Trees": GBTClassifier(featuresCol="features", labelCol="target", maxIter=30, seed=42)
    }

    evaluator_roc = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="target", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="target", metricName="f1")

    print("\n--- Model Benchmark Results ---")
    for name, clf in models.items():
        pipeline = Pipeline(stages=[assembler, scaler, clf])
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)

        roc_auc = evaluator_roc.evaluate(predictions)
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        print(f"[{name}] ROC-AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

    # 7. Cross-Validation on Random Forest
    print("\n--- Tuning Random Forest with 5-Fold Cross-Validation ---")
    rf = RandomForestClassifier(featuresCol="features", labelCol="target", seed=42)
    cv_pipeline = Pipeline(stages=[assembler, scaler, rf])

    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [30, 60, 100]) \
        .addGrid(rf.maxDepth, [4, 6, 8]) \
        .build()

    cv = CrossValidator(
        estimator=cv_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_roc,
        numFolds=5,
        seed=42
    )

    cv_model = cv.fit(train_data)
    best_preds = cv_model.transform(test_data)
    best_roc = evaluator_roc.evaluate(best_preds)
    best_acc = evaluator_acc.evaluate(best_preds)
    best_f1 = evaluator_f1.evaluate(best_preds)

    print(f"[Tuned Random Forest] ROC-AUC: {best_roc:.4f} | Accuracy: {best_acc:.4f} | F1: {best_f1:.4f}")

    # Feature importances
    best_rf_stage = cv_model.bestModel.stages[-1]
    importances = best_rf_stage.featureImportances.toArray()
    ranked_features = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print("\nTop 5 Clinical Risk Predictors:")
    for rank, (feat, score) in enumerate(ranked_features[:5], start=1):
        print(f"{rank}. {feat}: {score:.4f}")

    spark.stop()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
