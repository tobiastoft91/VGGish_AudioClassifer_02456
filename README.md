## Acoustic Scene Classification using transfer learning on VGGish pre trained model

By Tobias Toft Christensen, Mikkel Heber Hahn Petersen, and Anders Hansen Warming.

This repository is related to a paper presenting an acoustic scene classification method, which uses transfer learning on a VGGish pre-trained model. Transfer learning is a method where knowledge from solving one problem is gained and stored, and can subsequently be used and applied to a related problem. The performance of this method is evaluated on the TUT Acoustic Scenes 2017 data set. A data set collected in Finland by Tampere University of technology. The data collection has received funding from the European Research Council and is part of a DCASE \\textit{(Detection and Classification of Acoustic Scenes and Events)} 2017 challenge.
The project is related to the DTU course [*02456 Deep Learning*](http://kurser.dtu.dk/course/02456).

The project are written in Python programming language and some of the scripts is formatted into Jupyter Notebooks.
This repository contains a folder "tfRecordsReal" with the tfRecords for the training data, the validation data and the test data. Further more the resulting model is compiled and saved in the folder 3ClassModel and 15ClassModel, respectively for the 3 class and the 15 class classification problem. This repository also contains the pre-trained VGGish model (ref: https://github.com/tensorflow/models/tree/master/research/audioset).  
Finally this repository contain a Jupyter Notebook file (AcousticSceneClassifier.ipynb) with our model. 
