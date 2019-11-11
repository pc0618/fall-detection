# Project Title: FALL DETECTION WITH DEEP NEURAL NETS

Develop a model to perform human activity recognition, specifically to detect falls. Falls are an important health problem worldwide 
and reliable automatic fall detection systems can play an important role to mitigate negative consequences of falls.
The automatic detection of falls has attracted considerable attention in the computer vision and pattern recognition communities. There are two neural network models in the /src folder:
1) the Fall-Detection-with-CNNs-and-Optical-Flow based on the paper: "Vision-Based Fall Detection with Convolutional Neural Networks" by Núñez-Marcos
2) I3D models based on models reported in the paper: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" by Joao Carreira and Andrew Zisserman

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.
The repository contains the following files:
1. Source of downloading three Falling Datasets 
2. Dataset Pre-processing 
3. Source code for two models: a) optimal flow CNN b) I3D model
4. Results presentation. 

## Prerequisites and Installing
A step by step series of examples that tell you how to get a development env running
What things you need to install the software and how to install them:
For dataset preprocessing
```
pip install opencv-python
```
For model 1: the Fall-Detection-with-CNNs-and-Optical-Flow, check the file requirements.txt in the /scr folder for all the required dependencies
For model 2: follow the instructions for [installing Sonnet](https://github.com/deepmind/sonnet).
## Dataset description
There are three different datasets used in this project: 1)[UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html); 2)[Kinetic Human Action Video Dataset](https://deepmind.com/research/open-source/kinetics); 3)[Multiple Cameras Fall Dataset](http://www.iro.umontreal.ca/~labimage/Dataset/).

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```


![alt text](img/optical_flow_CNN.png)
![Sample Video of a person not falling](img/adl-01.mp4) ![Sample Video of a person slowing lying down](img/adl-40.mp4) ![Sample Video of a person falling](img/fall-01.mp4)

### And coding style tests

Explain what these tests test and why

```
Give an example
```

#### Inflated 3D Conv Net: Results

###### A. First Results (Possible Overlap in Train/Test Sets, See Addendum)

__Data__: RGB clips of 20 frames.

|       | Fall  | No Fall |
| :---  | :---  | :---    |
| Train |  40   |  200    |
| Test  |  18   |  15     |


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Hongyu Shen** 
* **William Wei** 
* **Asad Khan** 
* **Shirui Luo** 
* **Madhu Vellakal** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

