import random 


def get_user_choice():
    user_input = input("Enter Rock, Paper or Scissors : ")
    return user_input.lower()


def get_computer_choice():
    computer_choice = random.choice(['rock', 'paper', 'scissors'])
    return computer_choice


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


def play():
    user_input = get_user_choice()
    computer_choice = get_computer_choice()
    get_winner(user_input, computer_choice)

if __name__ == '__main__':
    play()