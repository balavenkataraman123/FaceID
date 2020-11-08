# FaceID
This is a face recognizer using CNN(couvloutional neural network). Add data of the face of every person in the family, and it can detect each person. useful for smart homes, as it can detect a person walking into  a room, and adjust the temperature and other things accordingly. it was my introduction to computer vision using neural networks. have a dataset in a folder with a sub folder with pictures with each person's face. train.py uses a haarcascade to figure out the precise location of the face in the training image, and use that data to make a CNN. Check.py uses the model, your webcam, opencv and haarcascade to classify the face as one of the faces from the dataset. 

How to use
python train.py data folder file path model name
