import cv2
from keras.models import load_model
import numpy as np
import time
import random


class CountdownTimer:
    def __init__(self, duration):
        self.duration = duration
        self.remaining_time = None

    def countdown_timer(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            prev_int_remaining_time = int(self.duration)
            self.remaining_time = None  # Initialize within the scope

            while True:
                elapsed_time = time.time() - start_time
                self.remaining_time = max(0, self.duration - elapsed_time)
                int_remaining_time = int(self.remaining_time)

                if int_remaining_time != prev_int_remaining_time:
                    print(f"Time remaining: {int_remaining_time} seconds")
                    prev_int_remaining_time = int_remaining_time

                if self.remaining_time <= 0:
                    break

                # Get and show the camera frame
                ret, frame = args[0].read()
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            return func(*args, **kwargs)  # Return the result of the wrapped function

        return wrapper


def get_computer_choice():
    computer_choice = random.choice(['rock', 'paper', 'scissors'])
    print("Computer chose:", computer_choice)
    return computer_choice


def get_prediction(frame, model):
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    prediction = model.predict(data)
    return prediction[0]


def get_winner(user_input, computer_choice):
    if user_input == computer_choice:
        print("It is a tie!")
    elif user_input == "rock" and computer_choice == "scissors":
        print("You win!")
    elif user_input == "paper" and computer_choice == "rock":
        print("You win!")
    elif user_input == "scissors" and computer_choice == "paper":
        print("You win!")
    else:
        print("Computer wins!")


def main():
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)

    # Set a larger window size for the camera feed
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 600)  # Set the desired width and height

    computer_wins = 0
    user_wins = 0
    rounds_to_win = 3

    timer = CountdownTimer(duration=3)

    while computer_wins < rounds_to_win and user_wins < rounds_to_win:
        timer = CountdownTimer(duration=3)

        @timer.countdown_timer
        def get_prediction_timed(cap, model):
            while True:
                ret, frame = cap.read()
                prediction = get_prediction(frame, model)
                predicted_class = np.argmax(prediction)
                predicted_class_name = {0: "rock", 1: "paper", 2: "scissors", 3: "nothing"}
                user_choice = predicted_class_name.get(predicted_class)

                print(f"You chose {user_choice}!")
                return user_choice  # Return the user choice from here

        user_input = get_prediction_timed(cap, model)
        computer_choice = get_computer_choice()
        get_winner(user_input, computer_choice)

        if user_input == computer_choice:
            print("It is a tie!")
        elif user_input == "rock" and computer_choice == "scissors":
            print("You win!")
            user_wins += 1
        elif user_input == "paper" and computer_choice == "rock":
            print("You win!")
            user_wins += 1
        elif user_input == "scissors" and computer_choice == "paper":
            print("You win!")
            user_wins += 1
        else:
            print("Computer wins!")
            computer_wins += 1

        print(f"Current Score: User {user_wins} - Computer {computer_wins}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
