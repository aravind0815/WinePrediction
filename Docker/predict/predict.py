import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(data_frame):
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Starting Spark Application")

    spark_session = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    spark_context._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    validation_data_path = str(sys.argv[1])
    trained_model_path = "/job/trainedmodel"

    validation_data_frame = (spark_session.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(validation_data_path))

    clean_validation_data = clean_data(validation_data_frame)

    prediction_model = PipelineModel.load(trained_model_path)

    predictions = prediction_model.transform(clean_validation_data)

    prediction_results = predictions.select(['prediction', 'label'])
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    model_accuracy = accuracy_evaluator.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model = {model_accuracy}')

    # F1 score computation using RDD API
    evaluation_metrics = MulticlassMetrics(prediction_results.rdd.map(tuple))
    weighted_f1_score = evaluation_metrics.weightedFMeasure()
    print(f'Weighted F1 Score of wine prediction model = {weighted_f1_score}')

    print("Exiting Spark Application")
    spark_session.stop()
