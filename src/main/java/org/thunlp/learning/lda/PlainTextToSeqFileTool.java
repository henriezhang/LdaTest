package org.thunlp.learning.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.thunlp.misc.Flags;
import org.thunlp.tool.GenericTool;

import java.io.IOException;

public class PlainTextToSeqFileTool implements GenericTool {
    Configuration conf = null;

    public PlainTextToSeqFileTool(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public void run(String[] args) throws Exception {
        Flags flags = new Flags();
        flags.add("input");
        flags.add("output");
        flags.parseAndCheck(args);

        convertToSequenceFile(
                new Path(flags.getString("input")),
                new Path(flags.getString("output")));
    }

    public void convertToSequenceFile(Path input, Path output)
            throws IOException, ClassNotFoundException, InterruptedException {
        Job job = new Job(this.conf);
        job.setJarByClass(this.getClass());
        job.setJobName("text-to-sequence-file");
        job.setMapperClass(ConvertMapper.class);
        job.setReducerClass(IdentityReducer.class);
        job.setNumReduceTasks(0);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        TextInputFormat.addInputPath(job, input);
        SequenceFileOutputFormat.setOutputPath(job, output);
        job.waitForCompletion(true);
    }

    public static class ConvertMapper extends Mapper<LongWritable, Text, Text, Text> {
        Text outkey = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            outkey.set(Long.toString(key.get()));
            context.write(outkey, value);
        }
    }
}
