# Fashion MNIST Classification 👕

This project uses Neural Networks to classify images from the Fashion MNIST dataset, which can be found in [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist).

## Dataset

The Fashion MNIST dataset consists of 70,000 grayscale images of 28x28 pixels, with 60,000 images for training and 10,000 images for testing. Each image is associated with a label indicating the type of clothing it represents, with a total of 10 classes:

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |


## Jupyter Notebook

The Jupyter Notebook in this repository, `fashion_mnist_classification.ipynb`, contains the code for loading the dataset, defining and training the model, and evaluating its performance. The notebook also includes visualizations of the training and validation loss and accuracy, as well as a confusion matrix to analyze the model's predictions.

## Requirements

To run the notebook, you need to have the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

To run the notebook, you can clone this repository and open the `fashion_mnist_classification.ipynb` file in Jupyter Notebook or JupyterLab. You also need to download the files from [Kaggle - Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist/code?datasetId=2243&sortBy=voteCount). Be sure to put the dataset in the same folder as the Jupyter Notebook and name the folder **data**.


## Results

The MLP achieves an accuracy of around 88% on the test set after 5 epochs of training. One way to improve the model could be by grouping the Shirts and Pullover labels and later create another model to classify specifically this two classes.

Another way to improve the current results is by adding more layers or increasing the number of epochs.
