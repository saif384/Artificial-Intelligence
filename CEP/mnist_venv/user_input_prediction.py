import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model("Handwritten_Digits.keras")

# Start a loop that continues until the user decides to exit
while True:
    # Prompt the user to input an image number or type "z" to quit
    user_input = input("Enter the image number you want to view (e.g., '1' for digit1.png) or type 'z' to quit: ")
    
    # Check if the user wants to exit
    if user_input.lower() == "z":
        print("Exiting the program.")
        break
    
    try:
        # Convert user input to an integer for the image number
        image_no = int(user_input)
    except ValueError:
        print("Please enter a valid integer.")
        continue  # Restart the loop for another input
    
    # Construct the file path for the selected image
    image_path = f"digits/digit{image_no}.png"
    
    # Check if the specified image exists
    if os.path.isfile(image_path):
        try:
            # Load the image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Check if the image was loaded properly
            if img is None:
                raise ValueError("Image not loaded properly")
            
            # Resize to 28x28 if necessary
            img = cv2.resize(img, (28, 28))
            
            # Invert, normalize, and reshape the image
            img = np.invert(img).astype("float32") / 255.0
            img = img.reshape(1, 28 * 28)
            
            # Make a prediction and display the result
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            
            # Display the image
            plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
            plt.show()
            
        except Exception as e:
            print(f"Error processing image {image_no}: {e}")
    else:
        print(f"No file found for image number {image_no}. Please make sure the file exists in the 'digits' folder.")

