import os
import sys
from termcolor import colored

sys.path.append('src')
from train_config import TrainConfig
from training import Training

class TrainingMenu:
    
    trainingConfig = TrainConfig('train_config.json')

    options = ["1. Start training", "2. Configuration", "3. Back"]
    choice = None

    def printMenu(self):
        os.system('cls')
        for option in self.options:
            print(colored(option, 'magenta'))

    def getOption(self):
        self.choice = input(colored("\n»Enter option: ", 'light_cyan'))

    def checkOption(self):
        if self.choice == "1":
            Training(self.trainingConfig)
            return True
        
        elif self.choice == "2":
            self.config()
            return True
        elif self.choice == "3":
            return False
        else:
            os.system('cls')
            print(colored("Invalid option", 'red'))
            input("Press any key to continue...")
            return True

    def config(self):

        configOptions = ["1. Epochs: ", "2. Learning Rate: ", "3. Batch Size: ", "4. Width x Height: ", "5. Channels: ", "6. Back"]
        
        while True:
            self.choice = None
            configParameters = [
                self.trainingConfig.get_epochs(),
                self.trainingConfig.get_learning_rate(),
                self.trainingConfig.get_batch_size(),
                self.trainingConfig.get_height(),
                self.trainingConfig.get_channels(),
                ""
            ]

            value = None
            configParameters = configParameters

            os.system('cls')
            print("Current configuration:")

            temp = 0
            for option in configOptions:
                print(colored(option, 'magenta'), colored(configParameters[temp], 'light_blue'))
                temp = temp + 1 
            
            self.choice = None
            self.getOption()
            
            if self.choice == "1":
                value = int(input(colored("Enter epochs [1, 100]: ", 'light_cyan')))
                if not self.trainingConfig.isInRange(value, 1, 100):
                    input("Press any key to continue...")
                else:
                    self.trainingConfig.set_epochs(value)
            
            elif self.choice == "2":
                value = float(input(colored("Enter learning rate [0.000001 - 0.01]: ",'light_cyan')))
                if not self.trainingConfig.isInRange(value, 0.000001, 0.01):
                    input("Press any key to continue...")
                else:
                    self.trainingConfig.set_learning_rate(value)
            
            elif self.choice == "3":
                value = int(input(colored("Enter batch size [8 - 128]: ",'light_cyan')))
                if not self.trainingConfig.isInRange(value, 8, 128):
                    input("Press any key to continue...")
                else:
                    self.trainingConfig.set_batch_size(value)

            
            elif self.choice == "4":
                value = int(input(colored("Enter width x height [64 - 144]: ",'light_cyan')))
                if not self.trainingConfig.isInRange(value, 64, 144):
                    input("Press any key to continue...")
                else:
                    self.trainingConfig.set_height(value)
                    self.trainingConfig.set_width(value)
            
            elif self.choice == "5":
                value = int(input(colored("Enter channels [1 - 3]: ",'light_cyan')))
                if not self.trainingConfig.isInRange(value, 1, 3):
                    input("Press any key to continue...")
                else:
                    self.trainingConfig.set_channels(value)
            
            elif self.choice == "6":
                break
            
            else:
                os.system('cls')
                print(colored("Invalid option", 'red'))
                input("Press any key to continue...")
                self.config()

            self.trainingConfig.save_config()

    def run(self):
        while True:
            self.printMenu()
            self.getOption()
            if not self.checkOption():
                break