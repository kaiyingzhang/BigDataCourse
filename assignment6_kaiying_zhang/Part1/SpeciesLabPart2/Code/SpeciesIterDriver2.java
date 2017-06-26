 
package U.CC;

 import org.apache.hadoop.fs.Path; 
  import org.apache.hadoop.fs.FileSystem; 
 import org.apache.hadoop.io.IntWritable; 
 import org.apache.hadoop.io.Text; 
 import org.apache.hadoop.mapred.JobClient; 
 import org.apache.hadoop.mapred.JobConf; 
 import org.apache.hadoop.mapred.Mapper; 
 import org.apache.hadoop.mapred.Reducer; 

import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
  
  
 public class SpeciesIterDriver2 { 
  
  public SpeciesIterDriver2(String in,int chain){

              String tempDir=in+ "\\part-00000";
     for(int i=0;i<chain;i++){
              JobClient client = new JobClient(); 
              JobConf conf = new JobConf(SpeciesIterDriver2.class); 
              conf.setJobName("Species Iter"); 
                
              conf.setNumReduceTasks(5); 
                
              conf.setOutputKeyClass(Text.class); 
              conf.setOutputValueClass(Text.class); 

              String path1="xml_output_species_Iterations/output"+(i+1);
			  
              Path path2 = new Path(path1);

              FileInputFormat.setInputPaths(conf, new Path(tempDir));

                FileOutputFormat.setOutputPath(conf, path2);

              conf.setMapperClass(SpeciesIterMapper2.class); 
              conf.setReducerClass(SpeciesIterReducer2.class); 
              conf.setCombinerClass(SpeciesIterReducer2.class); 
                
              client.setConf(conf); 

                    try{
                      FileSystem dfs=FileSystem.get(path2.toUri(),conf);
                      if(dfs.exists(path2)){
                        dfs.delete(path2,true);
                      }
                      JobClient.runJob(conf); 
                  }catch(Exception ex){
                      ex.printStackTrace(); 
                  }
                  tempDir=path1;
    }
    String inputview="output_species_Iterations/output"+ (chain)+"\\part-00000";
    SpeciesViewerDriver viewer=new SpeciesViewerDriver(inputview);
 } 
 } 
 