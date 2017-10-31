The source of this readme can be found here: https://upc-mai-dl.github.io/rnn-lab-guided/

## Time Series Regression (Wind Speed Prediction)

The goal of this example is to predict the wind speed of a geographical site given a window of the previous measurements. 
The data for this example has been extracted from the NREL Integration National Dataset Toolkit. 
The NREL dataset includes metereological information for more than 126,000 sites across the USA for the years 2007-2013 every 5 minutes.
The dataset included with this task has data for 4 sites and includes the variables wind speed at 100m, air density, temperature and air pressure. The original data is sampled every 5 minutes, to reduce computational cost the dataset is sampled every 15 minutes.
For this task we are going to use only the wind speed variable for one site. You will use the rest of the data during the autonomous laboratory. The data and the code are in the \Wind directory. The file Wind.npz contains the data matrices for all four sites. The data is in npz numpy format, this means that if you load the data from the file you will have an object that stores all the data matrices. This object has the attribute file that tells you the name of the matrices. We are going to use the matrix 90-45142.
The code of the example is in the WindPrediction.py file. This code reads the matrix of the first site and splits the data in a training set, a validation set and a test set. The script has a flag --verbose for verbose output, a flag --gpu for using an implementation for the recurrent layer suitable for gpu and a flag --config for the configuration file.7
Only the first variable (wind speed) is used. The data matrix is obtained generating windows of size lag+1 moving the window one step at a time. The first lag
columns of the matrix are the input attributes and the value to predict is the last one.
Because the recurrent layer needs its input as a 3-D matrix, the data matrix is transformed from 2-D to 3-D to obtain the shape (examples, sequence, attributes), in this case the sequence has size lag
and attributes has size 1.

The architecture for this task is composed by:

    * A first RNN layer with input size the length of the training window with an attribute per element in the sequence.
    * Optionally several stacked RNN layers
    * A dense layer with one neuron with linear activation, that is the one that computes the regression as output.

All the configuration for the network, the training and the split of the data is read from a configuration file in JSON format like this:
''
{
  "datasize": 0000000,
  "testsize": 000000, # Half validation, half test
  "lag": 0,
  "neurons": 000,
  "drop": 0.0,
  "nlayers": 0,
  "activation": "relu", # relu, tanh, sigmoid
  "activation_r": "sigmoid",
  "rnn": "LSTM", # LSTM, GRU
  "batch": 0000,
  "epochs": 00
}
''
The optimizer used is RMSprop (Keras documentation recommends it for for RNN), the loss function is the mean square error (MSE).
In this problem we can use as baseline the MSE of the persistence model, that is, predicting the t+1
step in the series as the value of the step t.

Elements to play with:
    * The size of the windows
    * The type of RNN (LSTM, GRU, SimpleRNN)
    * The dropout
    * The number of layers
    * Batch size
    * Number of epochs
    * Use an adaptive optimizer like adagrad or adam
