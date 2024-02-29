import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the pre-trained Keras model
model = load_model("har_model.h5")

# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assuming input size expected by your model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of the input image
def predict_class(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    # class_index = np.argmax(prediction)
    # return class_index

    class_indices = {'cycling': 0,'dancing': 1,'eating': 2,'laughing': 3,'listening_to_music': 4,'running': 5,'sleeping': 6,'using_laptop': 7}
    inv_class_indices = {v: k for k, v in class_indices.items()}
    predicted_class = inv_class_indices[np.argmax(prediction)]
    return predicted_class

# Function to handle the "Browse" button click event
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_image(file_path)

# Function to display the selected image on the GUI
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Adjust the size for display
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    class_label.config(text=f"Predicted Class: {predict_class(image_path)}")


# GUI setup
root = tk.Tk()
root.title("Image Classification App")

text_label = tk.Label(root, text="Human Activity Recognition", font=("Helvetica", 30,"bold"), fg="red")
text_label.pack(pady=20)

# Browse Button
browse_button = tk.Button(root, text="Upload Image", command=browse_image, font=("Helvetica", 16))
browse_button.pack(pady=40)

# Image Panel
panel = tk.Label(root)
panel.pack()

# Predicted Class Label
class_label = tk.Label(root, text="", font=("Helvetica", 18))
class_label.pack(pady=20)

# root.attributes('-topmost', True)  # Open the window on top
root.state('zoomed')
root.mainloop()
