Hello kind grader, code for this project can be found here: https://github.com/bjs250/CS7641-PS3

It's written to be compatible with Python Version 3.8.0, and you can install the libraries to run it in your virtual
environment from reqs.txt

The code is quite disorganized. Each method has its own file (e.g. pca.py), and within the file, functionality is controlled by boolean switches.

The data directory contains the datasets used for this analysis
The figures directory contains some of the figures used throughout the report
The params directory contains pickled data structures for the NN tuning parameters

part3.py contains an extensive set of switches used for generating the results of each of the 16 fold experiments.

The following resources were used for building the code and/or intuition for the experiments run in this project

References:

V-Measure
https://www.geeksforgeeks.org/ml-v-measure-for-evaluating-clustering-performance/
http://www1.cs.columbia.edu/~amaxwell/pubs/v_measure-emnlp07.pdf

k-means clustering
https://www.machinecurve.com/index.php/2020/04/16/how-to-perform-k-means-clustering-with-python-in-scikit/#what-is-k-means-clustering
https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

Expectation Maximization
https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
https://machinelearningmastery.com/expectation-maximization-em-algorithm/

PCA
https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

ICA
https://github.com/akcarsten/Independent_Component_Analysis
https://stats.stackexchange.com/questions/375124/what-does-ica-return
https://www.macroption.com/kurtosis-values/#:~:text=Kurtosis%20can%20reach%20values%20from%201%20to%20positive%20infinite.&text=A%20distribution%20that%20is%20more,more%20peaked%20and%20fatter%20tails).&text=Such%20distribution%20is%20called%20platykurtic%20or%20platykurtotic.
https://arxiv.org/pdf/0909.3052.pdf

FA
https://notebook.community/aborgher/Main-useful-functions-for-ML/ML/Dimensionality_reduction

Clustering with DR
https://stats.stackexchange.com/questions/232500/how-do-i-know-my-k-means-clustering-algorithm-is-suffering-from-the-curse-of-dim
