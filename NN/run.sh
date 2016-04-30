javac -cp libraries/mtj-1.0.jar -sourcepath src/nn -d compiled src/nn/*.java
java -cp compiled:libraries/mtj-1.0.jar nn/Main