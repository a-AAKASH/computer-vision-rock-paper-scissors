import cv2
from keras.models import load_model
import numpy as np
import time


def get_prediction(frame, model):
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    prediction = model.predict(data)
    return prediction[0]


def countdown_timer(duration):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            prev_int_remaining_time = int(duration) # Initialising the full duration

            while True:
                elapsed_time = time.time() - start_time
                remaining_time = max(0, duration - elapsed_time)
                int_remaining_time = int(remaining_time)
                
                if int_remaining_time != prev_int_remaining_time:
                    print(f"Time remaining: {int_remaining_time} seconds")
                    prev_int_remaining_time = int_remaining_time              
                    
                if remaining_time <= 0:
                    break

                # Get and show the camera frame
                ret, frame = args[0].read()
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            func(*args, **kwargs)
        return wrapper
    return decorator


def main():
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)

    # Set a larger window size for the camera feed
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 600)  # Set the desired width and height

    @countdown_timer(duration=3)  # Set the countdown duration here
    def get_prediction_timed(cap, model):
        while True:
            ret, frame = cap.read()
            prediction = get_prediction(frame, model)
            predicted_class = np.argmax(prediction)
            predicted_class_name = {0: "Rock", 1: "Paper", 2: "Scissors", 3: "Nothing"}

            print(f"You chose {predicted_class_name.get(predicted_class)}!")
            break  # Exit the loop after getting the prediction

    get_prediction_timed(cap, model)  # Run the prediction function with countdown

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
