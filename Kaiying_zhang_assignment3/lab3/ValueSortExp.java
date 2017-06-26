package Cloud.ApacheLog;

import java.nio.ByteBuffer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.IntWritable.Comparator;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class ValueSortExp {
 
 public static class MapTask extends
   Mapper<LongWritable, Text, IntWritable, Text> {
  public void map(LongWritable key, Text value, Context context)
    throws java.io.IOException, InterruptedException {
   String line = value.toString();
   String[] tokens = line.split(","); // This is the delimiter between
   String keypart = tokens[0].toString();
   System.out.println("Here");
   int valuePart = Integer.parseInt(tokens[1].replaceAll("\\s+",""));
   context.write(new IntWritable(valuePart), new Text(keypart));

  }
 }

 public static class ReduceTask extends
   Reducer<IntWritable, Text, Text, IntWritable> {
  public void reduce(IntWritable key, Iterable<Text> list, Context context)
    throws java.io.IOException, InterruptedException {
   
   for (Text value : list) {
    
    context.write(new Text(value),key);
    
   }
   
  }
 }

  public ValueSortExp(String str1, String str2) throws Exception {


	 String path = str1 + "\\part-00000";
   Path inputPath = new Path(path);
   Path outputDir = new Path(str2);

  // Create configuration
  Configuration conf = new Configuration(true);

  // Create job
  Job job = new Job(conf, "Test HIVE commond");
  job.setJarByClass(ValueSortExp.class);

  // Setup MapReduce
  job.setMapperClass(ValueSortExp.MapTask.class);
  job.setReducerClass(ValueSortExp.ReduceTask.class);
  job.setNumReduceTasks(1);

  // Specify key / value
  job.setMapOutputKeyClass(IntWritable.class);
  job.setMapOutputValueClass(Text.class);
  job.setOutputKeyClass(IntWritable.class);
  job.setOutputValueClass(IntWritable.class);
  job.setSortComparatorClass(IntComparator.class);
  // Input
  FileInputFormat.addInputPath(job, inputPath);
  job.setInputFormatClass(TextInputFormat.class);

  // Output
  FileOutputFormat.setOutputPath(job, outputDir);
  job.setOutputFormatClass(TextOutputFormat.class);

  /*
   * // Delete output if exists FileSystem hdfs = FileSystem.get(conf); if
   * (hdfs.exists(outputDir)) hdfs.delete(outputDir, true);
   * 
   * // Execute job int code = job.waitForCompletion(true) ? 0 : 1;
   * System.exit(code);
   */

  // Execute job
  int code = job.waitForCompletion(true) ? 0 : 1;
  System.exit(code);

 }
 
}