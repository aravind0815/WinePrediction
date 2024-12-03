import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def preprocess_data(data_frame):

    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

def configure_s3(spark_session):

    hadoop_conf = spark_session._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

if __name__ == "__main__":
    print("Starting Spark Application")

    spark_session = SparkSession.builder \
        .appName("WineQualityValidation") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    configure_s3(spark_session)

    if len(sys.argv) < 2:
        print("Usage: <script> <validation_data_path>")
        sys.exit(1)
    
    validation_data_path = sys.argv[1]
    trained_model_path = "s3://wineprecdit/trainedmodel" 

    print(f"Loading validation data from {validation_data_path}")
    raw_data = spark_session.read.format("csv") \
        .option("header", "true") \
        .option("sep", ";") \
        .option("inferschema", "true") \
        .load(validation_data_path)

    validation_data = preprocess_data(raw_data)

    print(f"Loading trained model from {trained_model_path}")
    prediction_model = PipelineModel.load(trained_model_path)

    print("Generating predictions")
    predictions = prediction_model.transform(validation_data)

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                                           predictionCol="prediction", 
                                                           metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)
    print(f"Test Accuracy of wine prediction model = {accuracy:.2f}")

    print("Computing Weighted F1 Score")
    prediction_results = predictions.select(['prediction', 'label'])
    prediction_metrics = MulticlassMetrics(prediction_results.rdd.map(tuple))
    weighted_f1_score = prediction_metrics.weightedFMeasure()
    print(f"Weighted F1 Score of wine prediction model = {weighted_f1_score:.2f}")

    print("Exiting Spark Application")
    spark_session.stop()
