package edu.cmu.ark;

import java.util.*;

public class App {
    static public void main(String[] args) {
        String buf;
        boolean printVerbose = false;
        String modelPath = null;

        List<Question> outputQuestionList = new ArrayList<>();
        boolean preferWH = false;
        boolean doNonPronounNPC = false;
        boolean doPronounNPC = true;
        Integer maxLength = 1000;
        boolean downweightPronouns = false;
        boolean avoidFreqWords = false;
        boolean dropPro = true;
        boolean justWH = false;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--debug"))
                GlobalProperties.setDebug(true);
            else if (args[i].equals("--verbose"))
                printVerbose = true;
            else if (args[i].equals("--model")) { // ranking model path
                modelPath = args[i + 1];
                i++;
            } else if (args[i].equals("--keep-pro")) {
                dropPro = false;
            } else if (args[i].equals("--downweight-pro")) {
                dropPro = false;
                downweightPronouns = true;
            } else if (args[i].equals("--downweight-frequent-answers")) {
                avoidFreqWords = true;
            } else if (args[i].equals("--properties")) {
                GlobalProperties.loadProperties(args[i + 1]);
            } else if (args[i].equals("--prefer-wh")) {
                preferWH = true;
            } else if (args[i].equals("--just-wh")) {
                justWH = true;
            } else if (args[i].equals("--full-npc")) {
                doNonPronounNPC = true;
            } else if (args[i].equals("--no-npc")) {
                doPronounNPC = false;
            } else if (args[i].equals("--max-length")) {
                maxLength = Integer.parseInt(args[i + 1]);
                i++;
            }
            Question q = new Question();

            QuestionRanker qr = new QuestionRanker();
            qr.loadModel(modelPath);
        }
    }
}
