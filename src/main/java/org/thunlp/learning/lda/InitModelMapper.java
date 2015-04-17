package org.thunlp.learning.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.thunlp.tool.FolderReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Convert space-separated words to DocumentWritable.
 * Key in&out: document id(not used).
 * Value in: Space-separated words of a document.
 * Value out: A DocumentWritable contains exactly the input document.
 *
 * @author sixiance
 */
public class InitModelMapper extends Mapper<Text, Text, Text, DocumentWritable> {

    private static Logger LOG = Logger.getAnonymousLogger();
    Map<String, Integer> wordmap = null;
    DocumentWritable doc = new DocumentWritable();
    List<Integer> wordbuf = new ArrayList<Integer>();

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        String[] words = value.toString().split(" +");
        wordbuf.clear();
        for (int i = 0; i < words.length; i++) {
            Integer id = wordmap.get(words[i]);
            if (id == null) {
                continue;
            }
            wordbuf.add(id);
        }
        doc.setNumWords(wordbuf.size());
        for (int i = 0; i < wordbuf.size(); i++) {
            doc.words[i] = wordbuf.get(i);
        }
        context.write(key, doc);
    }

    protected void setup(Mapper.Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        try {
            wordmap = loadWordList(conf.get("wordlist"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setWordList(Hashtable<String, Integer> wordmap) {
        this.wordmap = wordmap;
    }

    /**
     * Load word to id mapping.
     */
    private Map<String, Integer> loadWordList(String wordFile)
            throws IOException {
        Hashtable<String, Integer> keymap = new Hashtable<String, Integer>();
        FolderReader reader = new FolderReader(new Path(wordFile));
        Text key = new Text();
        IntWritable value = new IntWritable();
        while (reader.next(key, value)) {
            keymap.put(key.toString(), value.get());
        }
        reader.close();
        reader.end();
        return keymap;
    }
}
