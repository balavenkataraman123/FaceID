import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

EPOCHS = 20
IMG_WIDTH = 100
IMG_HEIGHT = 100
NUM_CATEGORIES = #set number of categories
TEST_SIZE = 0.1


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    
    # Create a tuple of lists to store the answer
    ans = ([], [])
    count = 0

    print("Loading data...")
    # Iterate through directories
    for dir_ in os.listdir(data_dir):
        print("Loading images from folder " + str(dir_))
        temp = []
        # Get path of directory
        path = os.path.join(data_dir + os.sep + str(dir_))
        # Iterate through files in directory
        for f in os.listdir(path):
            # Open, resize and save image
            f1 = os.path.join(path + os.sep + f)
            img = cv2.imread(f1)

            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)

            for (x,y,w,h) in faces:
                if w >= 200 and h >= 200:
                    img2 = img[y:y+h, x:x+w]

                    img1 = cv2.resize(img2, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
                    ans[0].append(img1)
                    ans[1].append(dir_)
                    count += 1
        

    print("Loaded " + str(count) + " images from " + str(NUM_CATEGORIES) + " directories")
    # Return the answer
    return ans

def get_model():
    """
    Makes a cnn
    """

    # Create a model
    model = tf.keras.models.Sequential([
        # Convulution layer #1 - 48 filters, 3x3 kernel
        tf.keras.layers.Conv2D(
            48, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(
            48, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),        
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(
            24, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),        
        # Max pooling layer using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Flatten units
        tf.keras.layers.Flatten(),
        #feel free to add more layers if your computer is good enough and you have enough data to avoid overfit
        # Hidden layer #1
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        # Hidden layer #2
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dropout(0.62),
        # Hidden layer #3
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
