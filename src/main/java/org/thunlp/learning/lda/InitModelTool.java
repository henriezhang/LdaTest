package org.thunlp.learning.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.thunlp.misc.AnyDoublePair;
import org.thunlp.misc.Flags;
import org.thunlp.tool.FolderReader;
import org.thunlp.tool.FolderWriter;
import org.thunlp.tool.GenericTool;

import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;
import java.util.logging.Logger;

public class InitModelTool implements GenericTool {
    private static Logger LOG = Logger.getAnonymousLogger();

    public void run(String[] args) throws Exception {
        Flags flags = new Flags();
        flags.add("input");
        flags.add("output_docs");
        flags.add("output_nwz");
        flags.add("num_topics");
        flags.add("wordlist");
        flags.add("max_num_words");
        flags.add("min_df");
        flags.parseAndCheck(args);

        Path input = new Path(flags.getString("input"));
        Path tfdf = new Path(flags.getString("wordlist") + ".tf_df");
        Path wordlist = new Path(flags.getString("wordlist"));
        int maxNumWords = flags.getInt("max_num_words");
        int minDf = flags.getInt("min_df");

        makeWordList(input, tfdf);
        int numWords = selectWords(tfdf, wordlist, maxNumWords, minDf);

        initModel(
                input,
                new Path(flags.getString("output_docs")),
                new Path(flags.getString("output_nwz")),
                wordlist,
                flags.getInt("num_topics"),
                numWords
        );

    }

    /**
     * Load word list, make word to id mapping.
     * All words are first sorted by their TF*IDF value. The top maxNumWords words
     * are used for training. TF*IDF is a widely used method for selecting
     * informative words in Information Retrieval, see Wikipedia for a more
     * detailed explanation.
     * <p/>
     * Note: words started with an underscore '_' are always kept, and they are
     * not count as number of words. This is used for special purpose.
     *
     * @param maxNumWords How many words to keep for training, -1 means all.
     * @return number of words used.
     * @throws IOException
     */
    public int selectWords(Path tfdf, Path wordlist, int maxNumWords, int minDf)
            throws IOException {
        Map<String, WordFreq> wordCounts = loadWordFreq(tfdf);
        List<String> specialKeys = new LinkedList<String>();
        WordFreq total = wordCounts.get(WordListMapper.NUM_DOCS_STRING);
        if (total == null) {
            throw new RuntimeException("No number of docs key in the word list.");
        }

        List<AnyDoublePair<String>> weights =
                new ArrayList<AnyDoublePair<String>>();
        for (Entry<String, WordFreq> e : wordCounts.entrySet()) {
            if (e.getKey().startsWith("_")) {
                specialKeys.add(e.getKey());
                continue;
            } else if (e.getKey().equals(WordListMapper.NUM_DOCS_STRING)) {
                continue;
            }
            WordFreq wf = e.getValue();
            if (wf.df > minDf) {
                double weight = wf.tf / total.tf * Math.log((total.df / wf.df));
                weights.add(new AnyDoublePair<String>(e.getKey(), weight));
            }
        }
        Collections.sort(weights, new Comparator<AnyDoublePair<String>>() {
            public int compare(AnyDoublePair<String> o1, AnyDoublePair<String> o2) {
                return Double.compare(o2.second, o1.second);
            }
        });
        FolderWriter writer =
                new FolderWriter(wordlist, Text.class, IntWritable.class);
        Text key = new Text();
        IntWritable value = new IntWritable();
        if (maxNumWords == -1)
            maxNumWords = Integer.MAX_VALUE;
        int numWords = Math.min(maxNumWords, weights.size());
        for (int i = 0; i < numWords; i++) {
            key.set(weights.get(i).first);
            value.set(i);
            writer.append(key, value);
        }
        for (String specialKey : specialKeys) {
            key.set(specialKey);
            value.set(numWords);
            writer.append(key, value);
            numWords++;
        }
        writer.close();
        LOG.info("Load " + wordCounts.size() + " words, keep " + numWords);
        return numWords;
    }

    public Map<String, WordFreq> loadWordFreq(Path sqfile)
            throws IOException {
        Hashtable<String, WordFreq> keymap = new Hashtable<String, WordFreq>();
        FolderReader reader = new FolderReader(sqfile);
        Text key = new Text();
        Text value = new Text();
        while (reader.next(key, value)) {
            WordFreq wf = new WordFreq();
            String str = value.toString();
            int split = str.indexOf(' ');
            wf.tf = (double) Long.parseLong(str.substring(0, split));
            wf.df = (double) Long.parseLong(str.substring(split + 1));
            keymap.put(key.toString(), wf);
        }
        reader.close();
        return keymap;
    }

    public void makeWordList(Path input, Path output) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        Job job = new Job(conf);
        job.setJobName("EstimateWordFreqForLDA");

        job.setMapperClass(WordListMapper.class);
        job.setReducerClass(WordListReducer.class);
        job.setCombinerClass(WordListCombiner.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(20);

        SequenceFileInputFormat.addInputPath(job, input);
        SequenceFileOutputFormat.setOutputPath(job, output);
        job.waitForCompletion(true);
    }

    public void initModel(
            Path input,
            Path outputDocs,
            Path outputNwz,
            Path wordlist,
            int numTopics,
            int numWords) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path tmpNwz = new Path(outputNwz + "_tmp").makeQualified(fs);
        wordlist = wordlist.makeQualified(fs);
        System.err.println("33");
        conf.set("wordlist", wordlist.toString());
        conf.set("output.nwz", tmpNwz.toString());
        conf.setInt("num.topics", numTopics);
        conf.setInt("num.words", numWords);
        FileSystem.get(conf).mkdirs(tmpNwz);
        Job job = new Job(conf);
        System.err.println("44");
        job.setJobName("InitializeModelForLDA");
        job.setMapperClass(InitModelMapper.class);
        job.setReducerClass(InitModelReducer.class);
        job.setNumReduceTasks(20);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DocumentWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DocumentWritable.class);

        SequenceFileInputFormat.addInputPath(job, input);
        SequenceFileOutputFormat.setOutputPath(job, outputDocs);

        System.err.println("55");
        job.waitForCompletion(true);

        System.err.println("66");
        combineModelParam(tmpNwz, outputNwz);
        System.err.println("77");
        fs.delete(tmpNwz, true);
        System.out.println("Done");
    }

    private void combineModelParam(Path inputNwz, Path outputNwz)
            throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        conf.setBoolean("take.mean", false);
        Job job = new Job(conf);
        job.setJobName("CombineModelParametersForLDA");

        SequenceFileInputFormat.addInputPath(job, inputNwz);
        SequenceFileOutputFormat.setOutputPath(job, outputNwz);

        job.setMapperClass(IdentityMapper.class);
        job.setReducerClass(CombineModelParamReducer.class);
        job.setNumReduceTasks(20);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(WordInfoWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(WordInfoWritable.class);

        job.waitForCompletion(true);
    }

    private static class WordFreq {
        public double tf;
        public double df;
    }
}
