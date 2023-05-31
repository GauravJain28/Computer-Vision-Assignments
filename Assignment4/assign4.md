# Assignment 4
In this assignment, you must perform transfer learning for an image
classification task. You will have to train a VGG16 model on the provided
custom dataset.

Note: This assignment needs to be done in Pytorch.

1. from the torch.hub take a VGG16 model that is pre-trained on the
ImageNet dataset.
   refernce : https://pytorch.org/hub/pytorch_vision_vgg/
   PyTorch code to load the pre-trained model:
torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

2. Please take the last two digits of your roll no. Let's say it is X.
Then, let (Y= X%4) download the dataset group corresponding to your Y.
   Dataset link:
https://drive.google.com/file/d/11ugotdlwiLX7Y-lIuC2DYsDBMXyl6I9V/view?usp=share_link

3. Modify the final FC (fully connected) layers to match the number of
classes provided in your dataset group.

4. Perform training on this modified network.

5. For starters, select the optimizer and hyperparameters of your choice.

6. You are expected to train the model using the data provided in the
train folder, while training perform validation on the data provided in
the valid folder after the end of every training epoch. Once the training
is complete, report the accuracy on the test images provided in the test
folder. (Select the model with the best accuracy on the validation set for
testing on the test set).

7. As part of your first experiment, apply various regularization terms
while training and show improvement.