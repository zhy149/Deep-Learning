Important!! Please make sure all libraries including tensorflow, sklearn.metrics, numpy are the latest version.

1. Need to install tensorflow and sklearn.metrics
2. Run train.py directly by calling python ./train.py
3. Run test.py directly by calling python ./test.py testset.npy, testset is not fixed name.
4. Please have test set and all other datasets under the same directories of these python files.
5. If run the training code, please make sure the input dataset name is Robot_Trials_DL.npy and under the same folder of these python files.
6. The traning code will save 20% of the original input dataset to a file called testset.npy if needed for further use later, and it will save a new trained model to a new folder called "model_one_new"
