package org.thunlp.learning.lda;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Iterator;

public class WordListReducer extends Reducer<Text, Text, Text, Text> {
    Text outvalue = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        long tf = 0;
        long df = 0;
        for (Text item : values) {
            String value = item.toString();
            if (value.charAt(0) == 'd') {
                df += Long.parseLong(value.substring(1));
            } else if (value.charAt(0) == 't') {
                tf += Long.parseLong(value.substring(1));
            }
        }
        outvalue.set(tf + " " + df);
        context.write(key, outvalue);
    }
}
