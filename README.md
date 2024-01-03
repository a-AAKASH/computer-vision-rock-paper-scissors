# Computer Vision RPS
This is the classic Rock Paper Scissors game but implemented in the computer. So you play against it. The only thing you need is your camera along with your Rock, Paper and Scissors.

The aim of the project is to implement a simple computer vision model created from Teachable Machine and then implementing the logic in python. 

*manual_rps.py* on the repo is a manual Rock, Paper and Scissors game. It was created to create the initial logic in the code which could be used further down the game itself. 

# Installation 

`requirements.txt` file for dependencies

If you require some knowledge on installing packages from the file, this link from [stackoverflow](https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from) or this from [nknk](https://note.nkmk.me/en/python-pip-install-requirements/) could be helpful.


# Usage Instruction

There's two branches as of now
    - `main` has the code up to date with camera play rps game 
    - `backup` has the backup code for the same, which I use to make changes to the game

The `main` branch should fulfill the game requirements.

# File Structure

The main file to use and run is the `camera_rps.py` file. The remaining files in the repo have been explained per below:
- keras_model.h5 - The model used to train the game is stored here. It could be personalised from [Teachable Machine](https://teachablemachine.withgoogle.com/train) if you fancy
- labels.txt - It has the labels corressponding to the class (rock, paper, scissors or nothing)
- manual_rps.py - This has the manual game which was a mid point for development of the logic of this project
- requirements.txt
- RPS-Template.py - It has the template for a basic capture window that uses the Open CV (CV2) library

