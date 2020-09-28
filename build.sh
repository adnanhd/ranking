#!/usr/bin/env sh

set -eux
TARGET=question-generation.jar

rm -rf bin
mkdir -p bin

javac -classpath ".:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/arkref.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/commons-lang-2.4.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/commons-logging.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/junit-3.8.2.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/jwnl.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/stanford-parser-2008-10-26.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/supersense-tagger.jar:/home/adnanhd/Projects/Enocta/QuestionGeneration/lib/weka-3-6.jar:" -d bin src/edu/cmu/ark/*.java src/edu/cmu/ark/ranking/*.java src/edu/cmu/ark/tests/*.java

cd bin
jar xf ../lib/commons-logging.jar
jar xf ../lib/jwnl.jar
jar xf ../lib/junit-3.8.2.jar
jar xf ../lib/stanford-parser-2008-10-26.jar
jar xf ../lib/supersense-tagger.jar
jar xf ../lib/weka-3-6.jar
#jar xf ../lib/commons-math-2.1.jar
jar xf ../lib/commons-lang-2.4.jar
jar xf ../lib/arkref.jar

jar cf $TARGET *
cd ..

mv bin/$TARGET .



