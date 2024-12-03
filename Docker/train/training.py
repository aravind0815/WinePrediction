import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def preprocess_data(data_frame):

    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

def configure_spark_session(app_name):

    spark_session = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    spark_session._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    
    return spark_session

if __name__ == "__main__":
    print("Starting Spark Application")

    spark_session = configure_spark_session("WineQualityPrediction")

    # Paths for training data and model
    training_data_path = "TrainingDataset.csv" 
    trained_model_path = "/job/trainedmodel" 

    print(f"Reading training data from {training_data_path}")
    raw_data = spark_session.read.format("csv") \
        .option("header", "true") \
        .option("sep", ";") \
        .option("inferschema", "true") \
        .load(training_data_path)
    training_data = preprocess_data(raw_data)

    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']
    label_column = "quality"

    print("Setting up pipeline components")
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    label_indexer = StringIndexer(inputCol=label_column, outputCol="label")

    rf_classifier = RandomForestClassifier(labelCol="label",
                                            featuresCol="features",
                                            numTrees=150,
                                            maxDepth=15,
                                            seed=150,
                                            impurity="gini")

    pipeline = Pipeline(stages=[feature_assembler, label_indexer, rf_classifier])

    print("Training the model")
    trained_model = pipeline.fit(training_data)

    print("Evaluating the model")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                                   predictionCol="prediction", 
                                                   metricName="accuracy")
    accuracy = evaluator.evaluate(trained_model.transform(training_data))
    print(f"Training Accuracy: {accuracy:.2f}")

    print("Setting up cross-validation")
    param_grid = ParamGridBuilder() \
        .addGrid(rf_classifier.maxDepth, [5, 10]) \
        .addGrid(rf_classifier.numTrees, [50, 150]) \
        .addGrid(rf_classifier.impurity, ["entropy", "gini"]) \
        .build()

    cross_validator = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=param_grid,
                                      evaluator=evaluator,
                                      numFolds=2)

    print("Performing cross-validation")
    cv_model = cross_validator.fit(training_data)

    print(f"Saving the best model to {trained_model_path}")
    cv_model.bestModel.write().overwrite().save(trained_model_path)

    print("Spark Application completed successfully")
    spark_session.stop()
