package org.thunlp.tool;

import java.io.IOException;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;

public class FolderReader {
  private Path path;
  private FileSystem fs;
  private JobConf conf;
    String originalFS = "";

  private SequenceFile.Reader currentReader;
  private FileStatus [] parts;
  private int currentPart;

  public FolderReader( Path path ) throws IOException {
    JobConf conf = new JobConf();
      originalFS = conf.get("fs.default.name");  //backup original configuration
      fs = FileSystem.get(conf);
      try {
          if(fs.exists(path)) {
              init( path, FileSystem.get(conf), conf);
              return;
          }
      } catch(Exception ex) {
          System.out.println("not hdfs filesystem");
          System.err.println("not hdfs filesystem");
      }

      try {
          conf.set("fs.default.name", "file:///");           //change configuration to local file system
          fs = FileSystem.get(conf);
          if(fs.exists(path)) {
              init(path, FileSystem.get(conf), conf);
              return;
          }
      } catch(Exception ex) {
          System.out.println("not local filesystem");
          System.err.println("not local filesystem");
          System.exit(100);
      }
  }

  public FolderReader( Path path, FileSystem fs, JobConf conf )
  throws IOException {
    init( path, fs, conf);
  }

  private void init( Path path, FileSystem fs, JobConf conf ) 
  throws IOException{
    this.path = path;
    this.fs = fs;
    this.conf = conf;

    currentPart = -1;
    currentReader = null;
    Path [] paths = { path };

    parts = fs.listStatus(paths, new PathFilter(){
      public boolean accept(Path p) {
        return !p.getName().startsWith(".") && !p.getName().startsWith("_");
      }
    });
    nextPart();
  }
  
  public Class getKeyClass() {
    return currentReader.getKeyClass();
  }
  
  public Class getValueClass() {
    return currentReader.getValueClass();
  }

  public boolean next( Writable key, Writable value ) throws IOException {
    if ( currentReader == null ) 
      return false;
    while ( ! currentReader.next(key, value) ) {
      if ( ! nextPart() ) 
        return false;
    }
    return true;
  }

  public void close() throws IOException {
    if ( currentReader != null ) {
      currentReader.close();
      currentReader = null;
    }
  }

  private boolean nextPart() throws IOException {
    close();
    currentPart++;
    if ( currentPart >= parts.length ) {
      return false;
    }
    currentReader = 
      new SequenceFile.Reader(fs, parts[currentPart].getPath(), conf);
    return true;
  }

    public void end() {
        conf.set("fs.defaultFS", originalFS);           //restore original configuration
    }
}
