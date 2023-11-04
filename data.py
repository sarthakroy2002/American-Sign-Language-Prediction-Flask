# Import all libraries
import tensorflow
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image

# Dictionary to search the classes from
dict = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"a",11:"b",12:"c",13:"d",14:"e",15:"f",16:"g",
        17:"h",18:"i",19:"j",20:"k",21:"l",22:"m",23:"n",24:"o",25:"p",26:"q",27:"r",28:"s",29:"t",30:"u",31:"v",32:"w",
        33:"x",34:"y",35:"z"}

# Create a folder to save the images
save_folder = "data"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Capture real-time image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        image_name = f"captured_image.png"
        image_path = os.path.join(save_folder, image_name)
        cv2.imwrite(image_path, frame)
    cv2.imshow("Real-time Image", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Load the image from the folder.
img = cv2.imread(r"D:\Coding\Codathon\data\image.jpg")

# Convert the image to a NumPy array.
img = np.array(img)

# Resize the image to 64x64.
img = cv2.resize(img, (64, 64))

# Convert the image to a TensorFlow image.
img = image.img_to_array(img)

# Expand the dimensions of the image to add a batch dimension.
img = np.expand_dims(img, axis=0)

# Load the TensorFlow model.
model_dl = tensorflow.keras.models.load_model("model_dl.h5")

# Predict the class of the image.
classes = model_dl.predict(img, batch_size=1)

# Print the predicted class.
array = classes.tolist()
array=array[0]
max_classes = max(array)
max_classes_index = array.index(max_classes)
keys_list = list(dict.keys())
key=dict[keys_list[max_classes_index]]
print(f'The predicted image corresponds to "{key}"')
