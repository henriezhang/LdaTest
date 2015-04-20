package org.thunlp.learning.lda;

/**
 * Created by henriezhang on 2015/4/20.
 */

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Iterator;

public class IdentityReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterator<Text> values, Context context) throws IOException, InterruptedException {
        while (values.hasNext()) {
            context.write(key, values.next());
        }
    }
}
