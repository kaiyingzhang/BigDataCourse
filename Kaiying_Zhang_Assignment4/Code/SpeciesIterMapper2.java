 
package U.CC;
 import java.io.IOException; 
  
 import org.apache.hadoop.io.Writable; 
 import org.apache.hadoop.io.WritableComparable; 
 import org.apache.hadoop.mapred.MapReduceBase; 
 import org.apache.hadoop.mapred.Mapper; 
 import org.apache.hadoop.mapred.OutputCollector; 
 import org.apache.hadoop.mapred.Reporter; 
 import org.apache.hadoop.io.Text; 
  
  
 public class SpeciesIterMapper2 extends MapReduceBase implements Mapper<WritableComparable, Writable, Text, Text> { 
  
   public void map(WritableComparable key, Writable value, 
                   OutputCollector output, Reporter reporter) throws IOException { 
  
     String data = ((Text)value).toString();
     int index = data.indexOf(":"); 
     if (index == -1) { 
       return; 
     } 

     String toParse = data.substring(0, index).trim(); 
     String[] splits = toParse.split("\t");
    //  System.out.println("splits:---"+splits[0]);
     if(splits.length == 0) {
       splits = toParse.split(" ");
            if(splits.length == 0) {
               return;
            }
     }
     String pagetitle = splits[0].trim(); 
     String pagerank = splits[splits.length - 1].trim();
     
     double currScore = 0.0;
     try { 
        currScore = Double.parseDouble(pagerank); 
     } catch (Exception e) { 
        currScore = 0.5;
     } 

     data = data.substring(index+1); 
     String[] pages = data.split(" "); 
    try{
      // System.out.println("pages:---"+pages[2]);
    }catch(Exception e){
      
    }
     int numoutlinks = 0;
     int cn=0;
     if (pages.length == 0) { 
        numoutlinks = 1;
     } else {
       for (String page : pages) {
         if(page.length() > 0) { 
            numoutlinks = numoutlinks + 1;
         }
       } 
      //  System.out.println("numoutlinks:---"+numoutlinks); 
     }

     Text toEmit = new Text((new Double(.98 * currScore / numoutlinks)).toString()); 
     for (String page : pages) { 
       if(page.length() > 0) {
         output.collect(new Text(page), toEmit);   
         output.collect(new Text(page), new  Text(" " + pagetitle)); 
       }
     } 

     output.collect(new Text(pagetitle), new Text(".02"));
     output.collect(new Text(pagetitle), new Text(" " + data)); 

   } 
 } 
 