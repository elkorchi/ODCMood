import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# Load model from JSON file
json_file = open('models/ferv3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('models/ferv3.h5')

VIDEO_NAME = 'vid3.mp4'
CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

def main():
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture(PATH_TO_VIDEO)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

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
            # Display the resulting frame
            img = frame[:,:,0:3]
            boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            #print(boxes)
            for i in range(boxes.shape[0]):
                pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                
                #cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0])+int(boxes[i][2]),int(boxes[i][1])+int(boxes[i][3])), color=(0, 255, 0), thickness=2)
                cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                
                #roi_gray=gray_img[pt1,pt2]#cropping region of interest i.e. face area from  image
                roi_gray = gray_img[int(boxes[i][1]):int(boxes[i][1]) + int(boxes[i][2]), int(boxes[i][0]):int(boxes[i][0]) + int(boxes[i][3])]
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                
                #find max indexed array
                max_index = int(np.argmax(predictions))

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            resized_img = cv2.resize(frame, (1000, 700))
            cv2.imshow('Video', resized_img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()