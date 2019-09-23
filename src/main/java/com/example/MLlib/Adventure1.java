package com.example.MLlib;

import com.example.pojo.JavaDocument;
import com.example.pojo.JavaLabeledDocument;
import com.google.gson.Gson;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.RFormula;
import org.apache.spark.ml.feature.RFormulaModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class Adventure1 {

    private static final SparkSession spark = SparkSession.builder()
            .appName("Adventure")
            .master("local[4]")
            .config("spark.some.config.option", "some-value")
            .getOrCreate();

    private static final String INPUT_FILE = "D:/spark/inputFile/ml.csv";

    private static final List<StructField> FIELDS = Arrays.asList(
            DataTypes.createStructField("rack", DataTypes.StringType, false),
            DataTypes.createStructField("type", DataTypes.StringType, false),
            DataTypes.createStructField("waistCircumference", DataTypes.IntegerType, true),
            DataTypes.createStructField("hipCircumference", DataTypes.DoubleType, true)
    );

    private static final StructType SCHEMA = DataTypes.createStructType(FIELDS);

    private static final Dataset<Row> GIRL_DF = spark.read()
            .schema(SCHEMA)
            .option("inferSchema", "false")
            .option("header", "true")
            .csv(INPUT_FILE);

    private static void correlationMatrix() {
        final List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.sparse(4, new int[]{0, 3}, new double[]{1.0, -2.0})),
                RowFactory.create(Vectors.dense(4.0, 5.0, 0.0, 3.0)),
                RowFactory.create(Vectors.dense(6.0, 7.0, 0.0, 8.0)),
                RowFactory.create(Vectors.sparse(4, new int[]{0, 3}, new double[]{9.0, 1.0}))
        );

        final StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });


        final Dataset<Row> df = spark.createDataFrame(data, schema);
        final Row r1 = Correlation.corr(df, "features").head();
        System.out.println("Pearson correlation matrix:\n" + r1.get(0).toString());

        final Row r2 = Correlation.corr(df, "features", "spearman").head();
        System.out.println("Spearman correlation matrix:\n" + r2.get(0).toString());

    }

    private static void summarizer() {

        final List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.dense(2.0, 3.0, 5.0), 1.0),
                RowFactory.create(Vectors.dense(4.0, 6.0, 7.0), 2.0)
        );

        final StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
                new StructField("weight", DataTypes.DoubleType, false, Metadata.empty())
        });

        final Dataset<Row> df = spark.createDataFrame(data, schema);

        final Row result1 = df.select(Summarizer.metrics("mean", "variance")
                .summary(new Column("features"), new Column("weight")).as("summary"))
                .select("summary.mean", "summary.variance").first();
        System.out.println("with weight: mean = " + result1.<Vector>getAs(0).toString() +
                ", variance = " + result1.<Vector>getAs(1).toString());

        final Row result2 = df.select(
                Summarizer.mean(new Column("features")),
                Summarizer.variance(new Column("features"))
        ).first();
        System.out.println("without weight: mean = " + result2.<Vector>getAs(0).toString() +
                ", variance = " + result2.<Vector>getAs(1).toString());

    }

    private static void mlAdventure() {
        GIRL_DF.show(40, false);

        // `~`: Separate target and terms. In this example: target is `type`.
        // `.`: All columns except the target/dependent variable.
        // `+`: Concat terms; “+ 0” means removing the intercept (this means that the y-intercept of the line that we will fit will be 0).
        // `:`: Interaction (multiplication for numeric values, or binarized categorical values).
        // In this case we want to use all available variables except target variable `type` (the .) and also add in the interactions between waistCircumference and rack and hipCircumference and rack, treating those as new features:
        final RFormula supervised = new RFormula().setFormula("type ~ . + rack:waistCircumference + rack:hipCircumference");

        final RFormulaModel fittedRF = supervised.fit(GIRL_DF);
        final Dataset<Row> preparedDF = fittedRF.transform(GIRL_DF);

        preparedDF.show(40, false);

        final Dataset<Row>[] dfs = preparedDF.randomSplit(new double[]{0.7, 0.3});

        final Dataset<Row> trainDF = dfs[0];
        final Dataset<Row> testDF = dfs[1];

        trainDF.show(40, false);
        testDF.show(40, false);

        final LogisticRegression lr = new LogisticRegression();
        lr.setMaxIter(10).setRegParam(0.01).setLabelCol("label").setFeaturesCol("features");

        System.out.println(lr.explainParams());

        final LogisticRegressionModel fittedLR = lr.fit(trainDF);
        fittedLR.transform(trainDF).select("label", "prediction").show(40, false);

    }

    private static void crossValidator() {
        // Prepare training documents, which are labeled.
        final Dataset<Row> training = spark.createDataFrame(Arrays.asList(
                new JavaLabeledDocument(0L, "a b c d e spark", 1.0),
                new JavaLabeledDocument(1L, "b d", 0.0),
                new JavaLabeledDocument(2L, "spark f g h", 1.0),
                new JavaLabeledDocument(3L, "hadoop mapreduce", 0.0),
                new JavaLabeledDocument(4L, "b spark who", 1.0),
                new JavaLabeledDocument(5L, "g d a y", 0.0),
                new JavaLabeledDocument(6L, "spark fly", 1.0),
                new JavaLabeledDocument(7L, "was mapreduce", 0.0),
                new JavaLabeledDocument(8L, "e spark program", 1.0),
                new JavaLabeledDocument(9L, "a e c l", 0.0),
                new JavaLabeledDocument(10L, "spark compile", 1.0),
                new JavaLabeledDocument(11L, "hadoop software", 0.0)
        ), JavaLabeledDocument.class);

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        final Tokenizer tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words");
        final HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        final LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.01);
        final Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});

        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        final ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(hashingTF.numFeatures(), new int[]{10, 100, 1000})
                .addGrid(lr.regParam(), new double[]{0.1, 0.01})
                .build();

        // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
        // This will allow us to jointly choose parameters for all Pipeline stages.
        // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
        // is areaUnderROC.
        final CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(2)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

        // Run cross-validation, and choose the best set of parameters.
        final CrossValidatorModel cvModel = cv.fit(training);

        // Prepare test documents, which are unlabeled.
        final Dataset<Row> test = spark.createDataFrame(Arrays.asList(
                new JavaDocument(4L, "spark i j k"),
                new JavaDocument(5L, "l m n"),
                new JavaDocument(6L, "mapreduce spark"),
                new JavaDocument(7L, "apache hadoop")
        ), JavaDocument.class);

        // Make predictions on test documents. cvModel uses the best model found (lrModel).
        final Dataset<Row> predictions = cvModel.transform(test);
        for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }
    }

    private static void trainValidationSplit() {

        // `~`: Separate target and terms. In this example: target is `type`.
        // `.`: All columns except the target/dependent variable.
        // `+`: Concat terms; “+ 0” means removing the intercept (this means that the y-intercept of the line that we will fit will be 0).
        // `:`: Interaction (multiplication for numeric values, or binarized categorical values).
        // In this case we want to use all available variables except target variable `type` (the .) and also add in the interactions between waistCircumference and rack and hipCircumference and rack, treating those as new features:
        final RFormula supervised = new RFormula().setFormula("type ~ . + rack:waistCircumference + rack:hipCircumference");

        final RFormulaModel fittedRF = supervised.fit(GIRL_DF);
        final Dataset<Row> preparedDF = fittedRF.transform(GIRL_DF);

        // Prepare training and test data.
        final Dataset<Row>[] splits = preparedDF.randomSplit(new double[]{0.9, 0.1}, 12345);
        final Dataset<Row> training = splits[0];
        final Dataset<Row> test = splits[1];

        final LinearRegression lr = new LinearRegression();

        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // TrainValidationSplit will try all combinations of values and determine best model using
        // the evaluator.
        final ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0.1, 0.01})
                .addGrid(lr.fitIntercept())
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 1.0})
                .build();

        // In this case the estimator is simply the linear regression.
        // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        final TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(lr)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid)
                // 80% for training and the remaining 20% for validation
                .setTrainRatio(0.8)
                // Evaluate up to 2 parameter settings in parallel
                .setParallelism(2);

        // Run train validation split, and choose the best set of parameters.
        final TrainValidationSplitModel model = trainValidationSplit.fit(training);

        // Make predictions on test data. model is the model with combination of parameters
        // that performed best.
        model.transform(test)
                .select("features", "label", "prediction")
                .show();
    }

    private static void trainValidationSplit2() {
        // Prepare training documents, which are labeled.
        final Dataset<Row> training = spark.createDataFrame(Arrays.asList(
                new JavaLabeledDocument(0L, "a b c d e spark", 1.0),
                new JavaLabeledDocument(1L, "b d", 0.0),
                new JavaLabeledDocument(2L, "spark f g h", 1.0),
                new JavaLabeledDocument(3L, "hadoop mapreduce", 0.0),
                new JavaLabeledDocument(4L, "b spark who", 1.0),
                new JavaLabeledDocument(5L, "g d a y", 0.0),
                new JavaLabeledDocument(6L, "spark fly", 1.0),
                new JavaLabeledDocument(7L, "was mapreduce", 0.0),
                new JavaLabeledDocument(8L, "e spark program", 1.0),
                new JavaLabeledDocument(9L, "a e c l", 0.0),
                new JavaLabeledDocument(10L, "spark compile", 1.0),
                new JavaLabeledDocument(11L, "hadoop software", 0.0)
        ), JavaLabeledDocument.class);

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        final Tokenizer tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words");
        final HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        final LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.01);
        final Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});

        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        final ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(hashingTF.numFeatures(), new int[]{10, 100, 1000})
                .addGrid(lr.regParam(), new double[]{0.1, 0.01})
                .build();

        // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
        // is areaUnderROC.
        final Evaluator evaluator = new BinaryClassificationEvaluator()
                .setMetricName("areaUnderROC")
                .setRawPredictionCol("prediction")
                .setLabelCol("label");

        final TrainValidationSplit tvs = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.75)
                // Evaluate up to 2 parameter settings in parallel
                .setParallelism(2);

        // Run cross-validation, and choose the best set of parameters.
        final TrainValidationSplitModel tvsModel = tvs.fit(training);

        // Prepare test documents, which are unlabeled.
        final Dataset<Row> test = spark.createDataFrame(Arrays.asList(
                new JavaDocument(4L, "spark i j k"),
                new JavaDocument(5L, "l m n"),
                new JavaDocument(6L, "mapreduce spark"),
                new JavaDocument(7L, "apache hadoop")
        ), JavaDocument.class);

        // Make predictions on test documents. cvModel uses the best model found (lrModel).
        final Dataset<Row> predictions = tvsModel.transform(test);
        for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }

        final Dataset<Row> trainedDF = tvsModel.transform(training);
        System.out.println("trainDF");
        trainedDF.show(40, false);

        System.out.println(evaluator.evaluate(trainedDF));

        final PipelineModel pipelineModel = (PipelineModel) tvsModel.bestModel();
        final LogisticRegressionModel trainedLR = (LogisticRegressionModel) pipelineModel.stages()[2];
        final LogisticRegressionTrainingSummary summaryLR = trainedLR.summary();
        final Gson gson = new Gson();
        System.out.println(gson.toJson(summaryLR.objectiveHistory()));

    }

    public static void main(String[] args) {
        crossValidator();
    }

}
