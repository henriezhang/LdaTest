package org.thunlp.tool;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.GenericOptionsParser;
import org.thunlp.learning.lda.LdaTrainer;
import org.thunlp.learning.lda.ShowTopics;

/**
 * 运行jar的入口
 */
public class EntryPoint {
    public static void main(String[] argsAll) throws Exception {
        Configuration conf = new Configuration();
        conf.addResource(new Path("/data/tdwadmin/tdwenv/tdwgaia/etc/hadoop/core-site.xml"));
        conf.addResource(new Path("/data/tdwadmin/tdwenv/tdwgaia/etc/hadoop/hdfs-site.xml"));
        conf.addResource(new Path("/data/tdwadmin/tdwenv/tdwgaia/etc/hadoop/mapred-site.xml"));
        conf.addResource(new Path("/data/tdwadmin/tdwenv/tdwgaia/etc/hadoop/yarn-site.xml"));
        String[] args = new GenericOptionsParser(conf, argsAll).getRemainingArgs();

        if (args.length < 1) {
            System.out.println("usage: train showModel");
            return;
        }

        String command = args[0];
        String[] realargs = new String[args.length - 1];
        System.err.println("Begin args:");
        for (int i = 0; i < realargs.length; i++) {
            realargs[i] = args[i + 1];
            System.err.print(realargs[i]);
        }
        System.err.println("\nEnd args");

        GenericTool tool = null;
        if (command.equals("train")) {
            tool = new LdaTrainer(conf);
        } else if (command.equals("showModel")) {
            tool = new ShowTopics(conf);
        }
        tool.run(realargs);
    }
}
