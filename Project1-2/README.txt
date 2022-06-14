1. Put the unzipped data folder in the same directory of codes, train set folder called "train", valid set folder called "valid" and test set folder called "test" should be in the "data" folder.
2. Run train.py to train the model, and it will save the model every 10 epochs and the user can interrupt. The PATH that the model will be saved is the same directory where codes and data are stored.
3. Run test.py to test the model, and it will load the model just saved from step 2 to test the test set.
4. Note: If you need to run on CPU, please type "map_location=torch.device('cpu')" after the  load model path as the second argument to make program run correctly.
5. The "model" file is a trained model to be tested by test.py