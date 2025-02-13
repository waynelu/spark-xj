package com.r2.xj

import java.sql.{Timestamp, DriverManager}
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}
import java.time.format.DateTimeFormatter

import com.cloudera.sparkts._
import com.cloudera.sparkts.stats.TimeSeriesStatisticalTests

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._

import com.cloudera.sparkts.models.ARIMA
import org.apache.spark.mllib.linalg.Vectors

import org.elasticsearch.spark.sql._

/**
 * An example exhibiting the use of TimeSeriesRDD for loading, cleaning, and filtering stock ticker
 * data.
 */
object DataForecast {
  /**
   * Creates a Spark DataFrame of (timestamp, symbol, price) from a tab-separated file of stock
   * ticker data.
   */
  def loadObservationsFromCSV(sqlContext: SQLContext, path: String, tag: String): DataFrame = {
    val rowRdd = sqlContext.sparkContext.textFile(path).map { line =>
      val tokens = line.split(',')
      val formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss").withZone(ZoneId.systemDefault());
      val dt = ZonedDateTime.parse(tokens(0), formatter)
      val tag = tokens(1)
      val value = tokens(2).toDouble
      Row(Timestamp.from(dt.toInstant), tag, value)
    }.filter { line => line.toString.contains(tag) }
    val fields = Seq(
      StructField("timestamp", TimestampType, true),
      StructField("tag", StringType, true),
      StructField("value", DoubleType, true)
    )
    val schema = StructType(fields)
    sqlContext.createDataFrame(rowRdd, schema)
  }

  def loadObservationsFromES(sqlContext: SQLContext, path: String, tag: String): DataFrame = {
    val options = Map("es.read.field.include" -> "tagtime, tagid, value", "es.query" -> "?q=tagid:".concat(tag));
    val df = sqlContext.read.format("org.elasticsearch.spark.sql").options(options).load(path)
    return df;
    
    /*
    val fields = Seq(
      StructField("timestamp", TimestampType, true),
      StructField("tag", StringType, true),
      StructField("value", DoubleType, true)
    )
    val schema = StructType(fields)
    sqlContext.createDataFrame(rowRdd, schema)
    */
  }

  def loadObservationsFromPI(sqlContext: SQLContext, path: String, tag: String): DataFrame = {
    //val url = "jdbc:pioledb://r2win.ddns.net/Data Source=r2win.ddns.net; Integrated Security=SSPI"
    val url = "jdbc:pioledb://r2win.ddns.net/Data Source=r2win.ddns.net"
    val driverClassName = "com.osisoft.jdbc.Driver";
    Class.forName(driverClassName).newInstance();
    val connection = DriverManager.getConnection(url)
    val connection = DriverManager.getConnection(url, "Administrator", "")
    val pStatement = connection.prepareStatement("SELECT time, tag, value FROM piinterp2 WHERE timestep='5m' and time >='2016-10-06 00:00:00' and time<='2016-10-12 00:00:00' and tag like ?");
    pStatement.setString(1, tag);
    val resultSet = pStatement.executeQuery();


    val stream = new Iterator[String] {
      def hasNext = resultSet.next()
      def next() = resultSet.getString(1) + "," + resultSet.getString(2) + "," + resultSet.getString(3)
    }.toStream

    val rdd = sqlContext.sparkContext.parallelize(stream).filter { line => !line.toString.contains("null") }
    val rowRdd = rdd.map{ line =>
      val tokens = line.split(',')
      val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.0").withZone(ZoneId.systemDefault());
      val dt = ZonedDateTime.parse(tokens(0), formatter)
      val time = tokens(0)
      val tag = tokens(1)
      val value = tokens(2).toDouble
      Row(Timestamp.from(dt.toInstant), tag, value)
    }

    val fields = Seq(
      StructField("timestamp", TimestampType, true),
      StructField("tag", StringType, true),
      StructField("value", DoubleType, true)
    )
    val schema = StructType(fields)
    sqlContext.createDataFrame(rowRdd, schema)
}

  def main(args: Array[String]): Unit = {
    val tag = args(0)
        
    val conf = new SparkConf().setAppName("XJ Data Forecast")
    conf.set("spark.io.compression.codec", "org.apache.spark.io.LZ4CompressionCodec")
    conf.set("spark.es.nodes", "localhost")
    conf.set("spark.es.port", "9200")
    conf.set("spark.es.resource", "xj-data/data")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //val dataObs = loadObservationsFromCSV(sqlContext, "./data_201601/adata_301.csv", tag)
    //val dataObs = loadObservationsFromES(sqlContext, "xj-data/data", tag)
    val dataObs = loadObservationsFromPI(sqlContext, "", tag)
    //println(dataObs.count)
    //return
    

    // Create an daily DateTimeIndex over August and September 2015
    val zone = ZoneId.systemDefault()
    val dtIndex = DateTimeIndex.uniformFromInterval(
      //ZonedDateTime.of(LocalDateTime.parse("2016-01-01T00:00:00"), zone), //CSV and ES
      //ZonedDateTime.of(LocalDateTime.parse("2016-01-11T00:00:00"), zone), //CSV and ES
      ZonedDateTime.of(LocalDateTime.parse("2016-10-07T00:00:00"), zone), //PI
      ZonedDateTime.of(LocalDateTime.parse("2016-10-11T00:00:00"), zone), //PI
      //new MinuteFrequency(5))
      new HourFrequency(1))
      
    // Align the data on the DateTimeIndex to create a TimeSeriesRDD
    val dataTsrdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex, dataObs,
      "timestamp", "tag", "value") // CSV and PI
      //"tagtime", "tagid", "value") // ES

    // Cache it in memory
    dataTsrdd.cache()

    // Count the number of series (number of symbols)
    //println(dataTsrdd.count())

    // Impute missing values using linear interpolation
    val filledTsrdd = dataTsrdd.fill("linear")

    /* autocorrelation case
    // Compute return rates
    val returnRates = filled.returnRates()
    
    // Compute Durbin-Watson stats for each series
    val dwStats = returnRates.mapValues(TimeSeriesStatisticalTests.dwtest)

    println(dwStats.map(_.swap).min)
    println(dwStats.map(_.swap).max)
    */
    
    //slice base ts rdd to prepare forecasting
    //val startDateTime = ZonedDateTime.of(LocalDateTime.parse("2016-01-02T00:00:00"), zone) // CSV and ES
    //val endDateTime = ZonedDateTime.of(LocalDateTime.parse("2016-01-07T00:00:00"), zone) // CSV and ES
    val startDateTime = ZonedDateTime.of(LocalDateTime.parse("2016-10-07T00:00:00"), zone) // PI
    val endDateTime = ZonedDateTime.of(LocalDateTime.parse("2016-10-08T00:00:00"), zone) // PI
    val sampleTsrdd = filledTsrdd.slice(startDateTime, endDateTime) 
    
    val sampleTs = sampleTsrdd.findSeries(tag)
    //println("base data size: " + baseTs.toArray.length);
    //println(baseTs)
    var forecastData = Array[Double]()
    try {
      val arimaModel = ARIMA.fitModel(1, 0, 1, sampleTs)
      //println("coefficients: " + arimaModel.coefficients.mkString(","))
      val forecast = arimaModel.forecast(sampleTs, 72)
      //println("forecast data size: " + forecast.toArray.length);
      //println("base + forecast of next 90 observations: " + forecast.toArray.mkString(","))
    
      forecastData = forecast.toArray.slice(2, forecast.toArray.length)
    } catch {
      case e: org.apache.commons.math3.linear.SingularMatrixException => {
        forecastData = sampleTs.toArray.slice(0, 10)
      } 
      case e: org.apache.commons.math3.exception.TooManyEvaluationsException => {
        forecastData = sampleTs.toArray.slice(0, 10)
      } 
    } finally {
      val actualDateTime = dtIndex.toZonedDateTimeArray
      val actualData = filledTsrdd.findSeries(tag).toArray
      val sampleData = sampleTs.toArray
      var counter = 0;
      var i = 0;
      var output = ""
      val formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss").withZone(ZoneId.systemDefault());
      for (i <- 0 to (actualDateTime.length-1)) {
        var str = "\"" + actualDateTime(i).format(formatter) + "\"," + actualData(i)
        if (actualDateTime(i).format(formatter) >= startDateTime.format(formatter)) {
          str += "," + (if(counter<sampleData.length) sampleData(counter) else "null")
          str += "," + (if(counter<forecastData.length) forecastData(counter) else "null")
          counter = counter + 1;
        } else {
          str += ",null,null"
        }
        output += "["+str+"],"
      }
      println("["+output.dropRight(1)+"]");
    }
  }
}
