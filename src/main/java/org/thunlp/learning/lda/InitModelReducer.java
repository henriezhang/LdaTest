package org.thunlp.learning.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

/**
 * Assign random topics to each word in the document. Output the overall n(w,z)
 * to a sequence file when all keys reduced.
 * Key in&out: Document id, not used.
 * Value in: DocumentWritable.
 * Value out: DocumentWritable with all topics initialized randomly.
 *
 * @author sixiance
 */
public class InitModelReducer extends Reducer<Text, DocumentWritable, Text, DocumentWritable> {
    int numTopics = 0;
    int numWords = 0;
    int[][] nwz = null;
    String outputNwz = null;
    Random randomProvider = new Random();

    public void reduce(Text key, Iterable<DocumentWritable> values, Context context)
            throws IOException, InterruptedException {
        for (DocumentWritable doc : values) {
            for (int i = 0; i < doc.getNumWords(); i++) {
                int word = doc.words[i];
                int topic = randomProvider.nextInt(numTopics);
                doc.topics[i] = topic;
                int[] counts = nwz[word];
                if (counts == null) {
                    counts = new int[numTopics];
                    counts[topic]++;
                    nwz[word] = counts;
                } else {
                    counts[topic]++;
                }
            }
            context.write(key, doc);
        }
    }

    protected void setup(Reducer.Context context)
            throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        numTopics = conf.getInt("num.topics", 0);
        numWords = conf.getInt("num.words", 0);
        outputNwz = conf.get("output.nwz");
        nwz = new int[numWords][];
    }

    void saveModelParameters(int[][] nwz, SequenceFile.Writer writer)
            throws IOException {
        IntWritable key = new IntWritable();
        WordInfoWritable value = new WordInfoWritable(numTopics);
        int[] zeros = new int[numTopics];
        Arrays.fill(zeros, 0);
        for (int i = 0; i < nwz.length; i++) {
            key.set(i);
            int[] counts = nwz[i];
            if (counts == null)
                counts = zeros;
            for (int j = 0; j < numTopics; j++) {
                value.setTopicCount(j, counts[j]);
            }
            value.setIsPartial(true);
            writer.append(key, value);
        }
    }

    protected void cleanup(Reducer.Context context)
            throws IOException, InterruptedException {
        String partName = "part-" + Math.abs(randomProvider.nextInt());
        Configuration envConf = context.getConfiguration();
        SequenceFile.Writer writer = SequenceFile.createWriter(
                FileSystem.get(envConf),
                envConf,
                new Path(outputNwz + "/" + partName),
                IntWritable.class,
                WordInfoWritable.class);
        saveModelParameters(nwz, writer);
        writer.close();
    }
}
