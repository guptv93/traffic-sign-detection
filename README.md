## NYU-CV-Fall-2018

### Assignment 2: Traffic sign competition

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies. Pay attention that the version of torchvision is specifically **0.4.1**

```bash
pip install -r requirements.txt
```

#### Training and validating your model
Run the script `main.py` to train your model.

- By default the images are loaded and resized to 244 x 244 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.
- By default a validation set is split for you from the training set and put in `[datadir]/val_images`. See data.py on how this is done.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --model ensemble.pth --data [data_dir]
```

The code uses three checkpoint files. One for the DenseNet model, one for the MobileNet model and one for the Ensemble model. Download the `.pth` files from the below link and put them in the root folder before running `main.py` or `evaluate.py`.

Link to PTH files : https://drive.google.com/open?id=1mu37TaIeB4KFb1iSx2mBIfncl2ByDZFY