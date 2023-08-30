import cv2
from keras.models import load_model
import numpy as np

def get_prediction(frame, model):
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    prediction = model.predict(data)
    return prediction[0]  # Get the probabilities for classes

def main():
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        prediction = get_prediction(frame, model)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability class

        cv2.imshow('frame', frame)
        print("Predicted class:", predicted_class, "with probability:", prediction[predicted_class])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()