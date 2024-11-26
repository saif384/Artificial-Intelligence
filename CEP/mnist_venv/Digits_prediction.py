import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model("Handwritten_Digits.keras")

# Initialize the image number
image_no = 1

# Process each image in the "digits" folder
while os.path.isfile(f"digits/digit{image_no}.png"):
    try:
        # Load the image in grayscale
        img = cv2.imread(f"digits/digit{image_no}.png", cv2.IMREAD_GRAYSCALE) # simplifies the image to black and white
        
        # Check if the image was loaded properly
        if img is None:
            raise ValueError("Image not loaded properly")
        
        # Resize to 28x28 if necessary
        img = cv2.resize(img, (28, 28))
        
        # Invert, normalize, and reshape the image
        img = np.invert(img).astype("float32") / 255.0 # np.invert convert black portion to white and vice-versa
        img = img.reshape(1, 28 * 28)
        
        # Make a prediction and display the result
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") # np.argmax return digit with highest probability
        
        # Display the image
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.show()
        
    except Exception as e:
        print(f"Error processing image {image_no}: {e}")
    
    image_no += 1
