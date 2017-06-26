package Cloud.ApacheLog;
import java.nio.ByteBuffer;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Iterator;
 import java.util.*;  
 
 import org.apache.hadoop.mapreduce.Job;

import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.conf.*;  
import org.apache.hadoop.io.*;  
import org.apache.hadoop.mapred.*;  
import org.apache.hadoop.util.*;  
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.IntWritable.Comparator;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;



  
 public class IpCount {  
  
   public static class IpMapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable>
	{
      private static final Pattern ipPattern = Pattern.compile("^([\\d\\.]+)\\s");
     // Reusable IntWritable for the count
     private final static IntWritable one = new IntWritable(1);  
  
     public void map(LongWritable fileOffset, Text lineContents,OutputCollector<Text, IntWritable> output, Reporter reporter)throws IOException
	 {
     // apply the regex to the line of the access log
     Matcher matcher = ipPattern.matcher(lineContents.toString());
	
     if(matcher.find())
     {
      // grab the IP
      String ip = matcher.group(1);
	  
      // output it with a count of 1
      output.collect(new Text(ip),one);
     }
     }
	 
    }  
  
    public static class IpReducer extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> 
   {

       public void reduce(Text ip, Iterator<IntWritable> counts,
        OutputCollector<Text, IntWritable> output, Reporter reporter)
        throws IOException {
    
        int totalCount = 0;
		
		String str = ip.toString();
        str = str + ",";
    
     // loop over the count and tally it up
        while (counts.hasNext())
     {
      IntWritable count = counts.next();
       totalCount += count.get();
     }
	 
    if(totalCount>100){
    output.collect(new Text(str), new IntWritable(totalCount));
	}
      }
	  }
	  

	
	  
    public static void main(String[] args) throws Exception {  
        JobConf conf = new JobConf(IpCount.class);
        conf.setJobName("ip-count");
        
        conf.setMapperClass(IpMapper.class);
        
	
        conf.setMapOutputKeyClass(Text.class);
        conf.setMapOutputValueClass(IntWritable.class);
        
        conf.setReducerClass(IpReducer.class);
        
		
		// No. of reduce tasks, equals no. output file
		conf.setNumReduceTasks(1);
		
        // take the input and output from the command line
        FileInputFormat.setInputPaths(conf, new Path(args[0]));
        FileOutputFormat.setOutputPath(conf, new Path(args[1]));

        JobClient.runJob(conf);
        //conf.setOutputKeyComparatorClass(IntComparator.class); 
		
		ValueSortExp v = new ValueSortExp(args[1],args[2]);
		
    }  
 
 }

 