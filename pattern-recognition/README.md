
# Implementation of Stress Detectors based on Physiological Signals

This repository contains the implementation of two papers related to stress detection which use the WESAD dataset.

[1] Schmidt, Philip & Reiss, Attila & Duerichen, Robert & Marberger, Claus & Van Laerhoven, Kristof. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. 400-408. DOI: 10.1145/3242969.3242985.

[2] Siirtola, Pekka. (2019). Continuous Stress Detection Using the Sensors of Commercial Smartwatch. 1198–1201. DOI: 10.1145/3341162.3344831.


## Project Structure and Development Process

### Environment

There are two assumed paths to run this code:
- a path to the git project
- a path to the WESAD dataset (this resource can be downloaded from [https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/))

Be sure that these are assigned appropriately for your environment

### File Structure
/
- README.md
- WESAD/ - contains the files of the WESAD dataset
- paper/ - contains the pdf file of our comparison between the previous papers
- classifiers.py - contains the classifiers used by the papers (LDA, QDA, RF, DT, AB, kNN)
- feature_extractor.py - contains the functions to compute features from the physiological signals (EDA, ECG, EMG, RESP, TEMP, BVP, ACC)
- WesadDataManager.py - contains classes and functions to load data from the WESAD dataset to be used in our application
- paper-siirtola.py - implementation of the paper of Siirtola [2] which uses only signals of Empatica E4
- paper-schmidt.py - implementation of the paper of Schmidt et al. [1] which uses signals from Empatica E4 and RespiBAN

### Dependencies

The project depends on multiple python libraries and packages. All the code is 
written for Python3 and is using
- pandas
- numpy
- neurokit (see installation in [https://neurokit.readthedocs.io/en/latest/index.html](https://neurokit.readthedocs.io/en/latest/index.html))
- sklearn
- scipy.signal

It is necessary that these dependencies are installed before executing some file.

### Execution
The execution of the paper-*.py files trains the classifiers and computes the accuracies for the different sensor combinations, storing the results in .csv files. The execution is done in the following way
- Paper of Schmidt et al.

``python paper-schmidt.py``

- Paper of Siirtola

``python paper-siirtola.py -r 2``