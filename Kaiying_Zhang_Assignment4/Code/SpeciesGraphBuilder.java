// 
 // Author - Jack Hebert (jhebert@cs.washington.edu) 
 // Copyright 2007 
 // Distributed under GPLv3 
 // 
// Modified - Dino Konstantopoulos
// Distributed under the "If it works, remolded by Dino Konstantopoulos, 
// otherwise no idea who did! And by the way, you're free to do whatever 
// you want to with it" dinolicense
// 
package U.CC;

 import org.apache.hadoop.fs.Path; 
 import org.apache.hadoop.fs.FileSystem;
 import org.apache.hadoop.io.IntWritable; 
 import org.apache.hadoop.io.Text; 
 import org.apache.hadoop.*; 
 import org.apache.hadoop.mapred.*; 
 import org.apache.hadoop.mapred.JobClient; 
 import org.apache.hadoop.mapred.JobConf; 
 import org.apache.hadoop.mapred.Mapper; 
 import org.apache.hadoop.mapred.Reducer; 
 import org.apache.hadoop.mapred.SequenceFileInputFormat; 
  
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
  
  
 public class SpeciesGraphBuilder { 
  
   public static void main(String[] args)  throws Exception
{ 
     JobClient client = new JobClient(); 
     JobConf conf = new JobConf(SpeciesGraphBuilder.class); 
     conf.setJobName("WikiSpecies P/R Graph Builder"); 
    
	 jobConf.set("xmlinput.start", "<page>");
	 jobConf.set("xmlinput.end", "</page>");
	 
	 jobConf.set(
				"io.serializations",
				"org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
				
	 jobConf.setInputFormat(XmlInputFormat.class);


     conf.setMapperClass(SpeciesGraphBuilderMapper.class); 
	 
	 conf.setReducerClass(SpeciesGraphBuilderReducer.class); 
	 
     conf.setMapOutputKeyClass(Text.class);
     conf.setMapOutputValueClass(Text.class);

     String path="output_species_builder";
     Path args0=new Path(args[0]);
     Path args1=new Path(path);
     FileInputFormat.setInputPaths(conf, args0);
     FileOutputFormat.setOutputPath(conf, args1);

     try{
        FileSystem dfs=FileSystem.get(args1.toUri(),conf);
        if(dfs.exists(args1)){
          dfs.delete(args1,true);
        }
        JobClient.runJob(conf); 
     }catch(Exception ex){
        ex.printStackTrace(); 
     }
     SpeciesIterDriver2 driver=new SpeciesIterDriver2(path,5); 

   } 
 }  