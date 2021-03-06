
package U.CC;

 import org.apache.hadoop.fs.Path; 
 import org.apache.hadoop.fs.FileSystem;
 import org.apache.hadoop.io.IntWritable; 
 import org.apache.hadoop.io.*; 
 import org.apache.hadoop.io.Text; 
 import org.apache.hadoop.mapred.JobClient; 
 import org.apache.hadoop.mapred.JobConf; 

import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
  
  
 public class SpeciesViewerDriver { 
  
    public SpeciesViewerDriver(String in){
     JobClient client = new JobClient(); 
     JobConf conf = new JobConf(SpeciesViewerDriver.class); 
     conf.setJobName("Species Viewer"); 
  

     conf.setOutputKeyClass(FloatWritable.class); 
     conf.setOutputValueClass(Text.class); 
  
     String path1="xml_output_species_viewer";
     Path path2=new Path(path1);
     FileInputFormat.setInputPaths(conf, new Path(in));
     FileOutputFormat.setOutputPath(conf, path2);
  
     conf.setMapperClass(SpeciesViewerMapper.class); 
     conf.setReducerClass(org.apache.hadoop.mapred.lib.IdentityReducer.class); 

     try{
      FileSystem dfs=FileSystem.get(path2.toUri(),conf);
      if(dfs.exists(path2)){
         dfs.delete(path2,true);
      }
         JobClient.runJob(conf); 
      }catch(Exception ex){
         ex.printStackTrace(); 
      } 
   } 
 } 
 