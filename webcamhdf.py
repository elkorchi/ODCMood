from statistics import mode
import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image



###################################################################################################################################
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input


# parameters for loading data and images
#detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_my_smallerCNN.91-0.66.hdf5'
emotion_labels = get_labels('fer2013')
#emotion_labels = ('unhappy', 'unhappy', 'unhappy', 'happy', 'unhappy', 'happy', 'neutral')
#emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
#face_detection = load_detection_model(detection_model_path)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
emotion_classifier = tf.keras.models.load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
###############################################################################################################################################

VIDEO_NAME = 'clip.mp4'
CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Capture device. Usually 0 will be webcam and 1 will be usb cam.
#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(PATH_TO_VIDEO)


minsize = 25 # minimum size of face
threshold = [0.5, 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


sess = tf.Session()
with sess.as_default():
    tf.global_variables_initializer().run()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    while(True):
        ret, frame = video_capture.read()
        if not ret:
            break
        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the resulting frame
        img = frame[:,:,0:3]
        boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #print(boxes)
        for i in range(boxes.shape[0]):
            pt1 = (int(boxes[i][0]), int(boxes[i][1]))
            pt2 = (int(boxes[i][2]), int(boxes[i][3]))
            
            #cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0])+int(boxes[i][2]),int(boxes[i][1])+int(boxes[i][3])), color=(0, 255, 0), thickness=2)
            #cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            
            #roi_gray=gray_img[pt1,pt2]#cropping region of interest i.e. face area from  image
            roi_gray = gray_img[int(boxes[i][1]):int(boxes[i][1]) + int(boxes[i][2]), int(boxes[i][0]):int(boxes[i][0]) + int(boxes[i][3])]
            roi_gray=cv2.resize(roi_gray,(64,64))
            img_pixels = image.img_to_array(roi_gray)


            gray_face = preprocess_input(roi_gray, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)



            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            cv2.rectangle(frame, pt1, pt2, color)

            cv2.putText(frame, emotion_text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        
        resized_img = cv2.resize(frame, (1000, 700))
        bgr_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', resized_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()