package org.thunlp.learning.lda;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.thunlp.misc.Counter;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map.Entry;

public class WordListMapper extends Mapper<Text, Text, Text, Text> {
    public static String NUM_DOCS_STRING = " ";
    Text outkey = new Text();
    Text outvalue = new Text();
    Counter<String> wordfreq = new Counter<String>();

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] words = value.toString().split(" ");
        wordfreq.clear();
        for (String w : words) {
            wordfreq.inc(w, 1);
        }
        Iterator<Entry<String, Long>> iter = wordfreq.iterator();
        long numWords = 0;
        while (iter.hasNext()) {
            Entry<String, Long> entry = iter.next();
            outkey.set(entry.getKey());
            outvalue.set("d1");
            context.write(outkey, outvalue);
            outvalue.set("t" + entry.getValue());
            context.write(outkey, outvalue);
            numWords += entry.getValue();
        }
        outkey.set(NUM_DOCS_STRING);
        outvalue.set("d1");
        context.write(outkey, outvalue);
        outvalue.set("t" + numWords);
        context.write(outkey, outvalue);
    }
}


