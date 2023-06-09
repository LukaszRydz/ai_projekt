import os
from termcolor import colored

from .webcam import cameraApp
from .trainingMenu import TrainingMenu


class MainMenu:
    options = ["1. Training", "2. Camera App", "3. Exit"]
    choice = None

    def printMenu(self):
        os.system('cls')
        for option in self.options:
            print(colored(option, 'magenta'))

    def getOption(self):
        self.choice = input(colored("\n»Enter option: ", 'light_cyan'))

    def checkOption(self):
        if self.choice == "1":
            TrainingMenu().run()
            return True
        elif self.choice == "2":
            cameraApp()
            return True
        elif self.choice == "3":
            print(colored("""
================================
              Exit
            Goodbye!
===============================""", 'light_yellow'))
            return False
        else:
            os.system('cls')
            print(colored("Invalid option", 'red'))
            input("Press any key to continue...")
            return True
        
    def run(self):
        while True:
            self.printMenu()
            self.getOption()
            if not self.checkOption():
                break