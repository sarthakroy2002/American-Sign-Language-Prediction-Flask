from flask import Flask, render_template, Response, request
import cv2
import os
import tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

camera = cv2.VideoCapture(0)

dict = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"a",11:"b",12:"c",13:"d",14:"e",15:"f",16:"g",
        17:"h",18:"i",19:"j",20:"k",21:"l",22:"m",23:"n",24:"o",25:"p",26:"q",27:"r",28:"s",29:"t",30:"u",31:"v",32:"w",
        33:"x",34:"y",35:"z"}

def generate_frames():
  while True:
    success, frame = camera.read()
    if not success:
      break
    else:
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/video_feed')
def video_feed():
  return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
  success, frame = camera.read()
  if not success:
    return Response(status=500)
  
  # Create a folder to save the images
  save_folder = "data"
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)

  # Saves the current frame
  image_name = f"image.jpg"
  image_path = os.path.join('data', image_name)
  cv2.imwrite(image_path, frame)

  # Load the TensorFlow model
  model_dl = tensorflow.keras.models.load_model("model_dl.h5")

  # Predict the class of the image
  img = cv2.imread(image_path)
  img = np.array(img)
  img = cv2.resize(img, (64, 64))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)

  classes = model_dl.predict(img, batch_size=1)

  # Get the predicted class label
  array = classes.tolist()
  array = array[0]
  max_classes = max(array)
  max_classes_index = array.index(max_classes)
  keys_list = list(dict.keys())
  key = dict[keys_list[max_classes_index]]

  # Render the video feed and the predicted class label on the same page
  return render_template('index.html', video_feed=frame, predicted_class=key)

if __name__ == '__main__':
  
  # Stop debugging
  app.run(debug=False)
