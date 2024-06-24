# Point Transformer

This is an unofficial comprehensive implementation of Point Transformer, a pioneering deep-learning technique for 3D data processing introduced by Zhao et al. (2021) [1]. 

## Data

Please follow the instructions in the data folder in order to download the datasets and place them in the same folder.

## Unit Testing

Each `.py` file in this repository is designed to be unit-tested by directly running them. To run the unit tests for a specific file, simply execute the file directly. For example:

```
python data_loader.py
```

This will run the unit tests defined in `data_loader.py` and provide feedback on whether the code behaves as expected.

## Train

To train the Point Transformer model, you can use either of the following scripts:

- ``train_bis.py``: This script trains the model for approximately 9 hours and achieves a test accuracy of 91.28%.
- ``train.py``: This alternative script offers faster training, taking only around 4 hours, while achieving a comparable accuracy of 91.12%.

Both scripts handle the training process, including data loading, model initialization, training loop, and evaluation.

## Test

You may find test the pretrained model and its performance by running the Jupiter notebook `test.ipynb`. This will load the test data and the pretrained model and compute the overall accuracy and the confusion matrix on the test set.

## References

[1] Zhao, Hengshuang, et al. "Point transformer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
