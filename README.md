hello guys...this is an image enhancer tool that is tried to build just for fun.
there are few pictures in the data/hi_res you can download more and place pictures there
i used CelebFaces Attributes (CelebA) Dataset available on kaggle.com
-----------------------------------------------------------------------------------------
Process:
  * simply run train.py after installing the dependencies
  * train will take time and it will create generator.pth and discriminator.pth files
  * for testing put a picture of a person in root folder named as test.jpg
  * after this run inference.py file and see the results
-----------------------------------------------------------------------------------------
dependecies:
  * torch
  * torchvision
  * opencv
