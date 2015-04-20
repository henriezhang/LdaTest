package org.thunlp.learning.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.thunlp.misc.Flags;
import org.thunlp.tool.GenericTool;

import java.io.IOException;

/**
 * Perform Gibbs Sampling on a set of documents, according to the NWZ file.
 * First, we pass the documents to GibbsSamplingReducer by IdentityMapper. Then,
 * GibbsSamplingReducer do the sampling, output the documents with new topic
 * assignmentsm, and also output the changed NWZ file. Finally, another
 * map-reduce combines the NWZ files from different reducers into one.
 * <p/>
 * The reason of not doing sampling in the map stage is efficiency. We have to
 * load a possibly large NWZ file into memory before sampling, which may take a
 * lot of time. Normally Hadoop allocates one reducer and several mappers for
 * one machine. If we do the sampling in the map stage, the same NWZ-loading
 * work would be repeated several times on one machine, which is a waste of
 * resource and significantly slows down the whole training process.
 *
 * @author sixiance
 */
public class GibbsSamplingTool implements GenericTool {
    public static double RESOLUTION = 0.01;
    public Configuration conf = null;

    public GibbsSamplingTool(Configuration conf) {
        this.conf = conf;
    }

    public void run(String[] args) throws Exception {
        Flags flags = new Flags();
        flags.add("input_docs");
        flags.add("input_nwz");
        flags.add("output_docs");
        flags.add("output_nwz");
        flags.add("alpha");
        flags.add("beta");
        flags.add("num_topics");
        flags.add("num_words");
        flags.parseAndCheck(args);

        double likelihood = sampling(
                new Path(flags.getString("input_docs")),
                new Path(flags.getString("output_docs")),
                new Path(flags.getString("input_nwz")),
                new Path(flags.getString("output_nwz")),
                flags.getDouble("alpha"),
                flags.getDouble("beta"),
                flags.getInt("num_topics"),
                flags.getInt("num_words")
        );

        System.out.println("Done with likelihood " + likelihood);
    }

    /**
     * Do Gibbs Sampling on the document set, and return the overall likelihood.
     *
     * @param inputDocs
     * @param outputDocs
     * @param inputNwz
     * @param outputNwz
     * @param alpha
     * @param beta
     * @param numWords
     * @param numTopics
     * @return Overall likelihood.
     * @throws IOException
     */
    public double sampling(
            Path inputDocs, Path outputDocs,
            Path inputNwz, Path outputNwz,
            double alpha, double beta,
            int numTopics, int numWords)
            throws IOException, ClassNotFoundException, InterruptedException {
        FileSystem fs = FileSystem.get(this.conf);
        Path tmpNwz = new Path(outputNwz + "_tmp").makeQualified(fs);
        inputNwz = inputNwz.makeQualified(fs);
        fs.mkdirs(tmpNwz);

        conf.set("input.nwz", inputNwz.toString());
        conf.set("output.nwz", tmpNwz.toString());
        conf.set("alpha", Double.toString(alpha));
        conf.set("beta", Double.toString(beta));
        conf.set("num.topics", Integer.toString(numTopics));
        conf.set("num.words", Integer.toString(numWords));

        Job job = new Job(conf);
        job.setJobName("GibbsSamplingForLDA");
        SequenceFileInputFormat.addInputPath(job, inputDocs);
        SequenceFileOutputFormat.setOutputPath(job, outputDocs);

        job.setMapperClass(IdentityMapper.class);
        job.setReducerClass(GibbsSamplingReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DocumentWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DocumentWritable.class);

        int exitCode = job.waitForCompletion(true) ? 0 : 1;
        Counters counters = job.getCounters();

        double likelihood = counters.findCounter(GibbsSamplingCounter.LIKELIHOOD).getValue() / GibbsSamplingTool.RESOLUTION;

        combineModelParam(inputNwz, tmpNwz, outputNwz);
        fs.delete(tmpNwz, true);

        return likelihood;
    }

    private void combineModelParam(Path refNwz, Path inputNwz, Path outputNwz)
            throws IOException, ClassNotFoundException, InterruptedException {
        Job job = new Job(this.conf);
        job.setJobName("CombineModelParametersForLDA");
        job.setJarByClass(this.getClass());

        SequenceFileInputFormat.addInputPath(job, inputNwz);
        SequenceFileInputFormat.addInputPath(job, refNwz);
        SequenceFileOutputFormat.setOutputPath(job, outputNwz);

        job.setMapperClass(IdentityMapper.class);
        job.setReducerClass(CombineModelParamReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(WordInfoWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(WordInfoWritable.class);
        job.waitForCompletion(true);
    }

    public enum GibbsSamplingCounter {LIKELIHOOD}

}
