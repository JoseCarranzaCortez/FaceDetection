# Face Detection Engine through Neural networks
This is an experimenetal engine that detects faces on photographs using neural networks. It is important to note that this was developed with experimental purposes and has not been intended to be usedin real-life scenarios. 

# Dataset
The dataset used to train the neural network is the one provided by MIT on the following url:
http://cbcl.mit.edu/software-datasets/FaceData2.html

# Test Classes
There are to important classes on the project, that serve as entry points for the system.

### Training 
Training is done using the class test.TestTraining. Training is done through an input file with a specific format. The MIT dataset has already been formatted to follow this standard.

### Test
The current implementation of the classifier uses 2 neural network classifiers combined using an AND operation, to reduce false positives during classification. Such classifiers have already been trained and saved as files (as serialized Java Objects) and can be located on the directory data/classifier/1.perceptron and data/classifier/2.perceptron.

To run the classifier on a photograph, open a terminal on the proyect diretory and run the following command (LINUX Version): 

```sh
$ java -cp "build/classes/:lib/*" test.TestWindow -c1 ./data/classifiers/1.perceptron -c2 ./data/classifiers/2.perceptron -in <input_image_file_path> -out <output_image_file_path (with .jpg extension)> -t <classification_treshold>
```

On the previous command:
- input_image_file_path is the image in which we want to detect faces.
- output_image_file_path is where the output image will be saved. In this image all the found faces will be drawn. 
- classification_treshold is a number between 0 and 1 that determines the starting score (in terms of probability) with which the classifier classifies something as a face.

A real example:

```sh
$ java -cp "build/classes/:lib/*" test.TestWindow -c1 ./data/classifiers/1.perceptron -c2 ./data/classifiers/2.perceptron -in ./data/images/facial.jpg -out ./data/images/facial_processed.jpg -t 0.9
```

Another example:
```sh
$ java -cp "build/classes/:lib/*" test.TestWindow -c1 ./data/classifiers/1.perceptron -c2 ./data/classifiers/2.perceptron -in ./data/images/girl.jpg -out ./data/images/girl_processed.jpg -t 0.5
```

# Conclusion
This is a basic example that demostrates that a Multilayer Perceptron can learn how a face looks like, and can be used to find faces in photographs with little image pre-processing.

# Future Changes
During the following days the code will be completely translated to english, and some further optimizations will be made. 
