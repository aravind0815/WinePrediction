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

def configure_s3(spark_session):

    hadoop_conf = spark_session._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

if __name__ == "__main__":
    print("Initializing Spark Application")

    spark_session = SparkSession.builder \
        .appName("WineQualityPrediction") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    configure_s3(spark_session)

    training_data_path = "s3://winecluster/TrainingDataset.csv"  
    model_output_path = "s3://winecluster/trainedmodel" 

    print(f"Loading training data from {training_data_path}")
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

    print("Configuring data pipeline")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    indexer = StringIndexer(inputCol=label_column, outputCol="label")

    rf_classifier = RandomForestClassifier(labelCol="label",
                                            featuresCol="features",
                                            numTrees=100,
                                            maxDepth=10,
                                            seed=150)

    pipeline = Pipeline(stages=[assembler, indexer, rf_classifier])

    print("Training model")
    model = pipeline.fit(training_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                   predictionCol="prediction",
                                                   metricName="accuracy")
    accuracy = evaluator.evaluate(model.transform(training_data))
    print(f"Initial Model Accuracy: {accuracy:.2f}")

    print("Starting cross-validation")
    param_grid = ParamGridBuilder() \
        .addGrid(rf_classifier.numTrees, [50, 100]) \
        .addGrid(rf_classifier.maxDepth, [5, 10]) \
        .build()

    cross_validator = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=param_grid,
                                      evaluator=evaluator,
                                      numFolds=3)

    cv_model = cross_validator.fit(training_data)
    best_model = cv_model.bestModel

    print(f"Saving best model to {model_output_path}")
    best_model.write().overwrite().save(model_output_path)

    print("Application finished successfully")
    spark_session.stop()
