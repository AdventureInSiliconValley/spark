package com.example.sql;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class Adventure1 {

    private static final SparkSession spark = SparkSession.builder()
            .appName("Adventure")
            .master("local[4]")
            .config("spark.some.config.option", "some-value")
            .getOrCreate();

    private static final String INPUT_FILE = "D:/spark/inputFile/girls.csv";

    private static final List<StructField> FIELDS = Arrays.asList(
            DataTypes.createStructField("name", DataTypes.StringType, false),
            DataTypes.createStructField("age", DataTypes.IntegerType, false),
            DataTypes.createStructField("birthday", DataTypes.DateType, true),
            DataTypes.createStructField("cherryPoppedTime", DataTypes.TimestampType, true)
    );

    private static final StructType SCHEMA = DataTypes.createStructType(FIELDS);

    private static final Dataset<Row> GIRL_DF = spark.read()
            .schema(SCHEMA)
            .option("inferSchema", "false")
            .option("header", "true")
            .option("dateFormat", "yyyy:MM:dd")
            .option("timestampFormat", "MM/dd/yyyy HH:mm:ss.SSSZZ")
            .csv(INPUT_FILE);

    private static void test1() {

        GIRL_DF.createOrReplaceTempView("girlDF");

        // root
        // |-- name: string (nullable = true)
        // |-- age: integer (nullable = true)
        // |-- birthday: date (nullable = true)
        // |-- cherryPoppedTime: timestamp (nullable = true)
        GIRL_DF.printSchema();

        GIRL_DF.show();

        final Dataset<Row> resultDF = spark.sql("select * from girlDF where birthday > cast('2019-07-11' as date)");

        resultDF.show(40, false);

        final List<Row> rows = resultDF.collectAsList();
        System.out.println(rows);

        resultDF.coalesce(1).write().mode(SaveMode.Overwrite).csv("D:/spark/outputFile/girls.csv");

    }

    private static void test2() {

        // root
        // |-- name: string (nullable = true)
        // |-- age: integer (nullable = true)
        // |-- birthday: date (nullable = true)
        // |-- cherryPoppedTime: timestamp (nullable = true)
        GIRL_DF.printSchema();

        GIRL_DF.show(40, false);

        GIRL_DF.createOrReplaceTempView("girlDF");

        final Dataset<Row> resultDF = spark.sql("select * from girlDF where cherryPoppedTime > cast('2019-08-12 10:05:03' as timestamp)");

        resultDF.show(40, false);

        final List<Row> rows = resultDF.collectAsList();
        System.out.println(rows);
        writeFile(rows);

        resultDF.coalesce(1)
                .write()
                .mode(SaveMode.Overwrite)
                .option("dateFormat", "MM/dd/yyyy")
                .option("timestampFormat", "MM/dd/yyyy HH:mm:ss.SSSZZ")
                .csv("D:/spark/outputFile/girls2.csv");

    }

    private static void test3() {
        try {
            try (Stream<Path> paths = Files.walk(Paths.get("D:/spark/outputFile/girls2.csv"))) {
                paths
                        .filter(path -> {
                            // D:\spark\outputFile\girls2.csv   girls2.csv  true
                            System.out.println("filter: " + path + ", path.getFileName(): " + path.getFileName() + ", isRegularFile: " + Files.isRegularFile(path));
                            return Files.isRegularFile(path) && path.getFileName().toString().endsWith(".csv");
                        })
                        .forEach(path -> {
                            final String filePath = path.toString();
                            System.out.println("forEach path: " + filePath + ", path.getFileName(): " + path.getFileName());

                            readFile(filePath);
                        });
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void listFilesForFolder(final File folder) {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                System.out.println(fileEntry.getName());
            }
        }
    }

    /**
     * readFile
     *
     * @param fileName The name of the file to open.
     */
    public static void readFile(final String fileName) {

        // This will reference one line at a time
        String line = null;

        try {
            // FileReader reads text files in the default encoding.
            final FileReader fileReader =
                    new FileReader(fileName);

            // Always wrap FileReader in BufferedReader.
            final BufferedReader bufferedReader =
                    new BufferedReader(fileReader);

            while ((line = bufferedReader.readLine()) != null) {
                System.out.println("read: " + line);
            }

            // Always close files.
            bufferedReader.close();
        } catch (FileNotFoundException ex) {
            System.out.println(
                    "Unable to open file '" + fileName + "'");
        } catch (IOException ex) {
            System.out.println(
                    "Error reading file '" + fileName + "'");
        }

    }

    public static void writeFile(final List<Row> rows) {
        // The name of the file to open.
        final String fileName = "D:/spark/outputFile/temp.txt";

        try {
            // Assume default encoding.
            final FileWriter fileWriter =
                    new FileWriter(fileName);

            // Always wrap FileWriter in BufferedWriter.
            final BufferedWriter bufferedWriter =
                    new BufferedWriter(fileWriter);

            // Note that write() does not automatically
            // append a newline character.
            bufferedWriter.write("Hello there,");
            bufferedWriter.write(" here is some text.");
            bufferedWriter.newLine();
            bufferedWriter.write("We are writing");
            bufferedWriter.write(" the text to the file.");
            bufferedWriter.newLine();

            rows.forEach(row -> {
                try {
                    bufferedWriter.write(row.getString(0) + "is " + row.getInt(1) + " years old, and her birthday is " + row.getDate(2) + ". Her cherry was popped at " + row.getTimestamp(3) + ".");
                    bufferedWriter.newLine();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            // Always close files.
            bufferedWriter.close();
        } catch (IOException ex) {
            System.out.println("Error writing to file '" + fileName + "'");
        }
    }

    public static void main(String[] args) {

        test1();

    }

}
