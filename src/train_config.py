import json
from termcolor import colored

class TrainConfig:
    def __init__(self, config_name):
        self.config_name = './' + config_name
        self.data = self.load_config()

        self.train_parameters = self.data['trainParameters']
        
        self.epochs = self.train_parameters['epochs']
        self.learning_rate = self.train_parameters['learningRate']
        self.batch_size = self.train_parameters['batchSize']
        self.width = self.train_parameters['width']
        self.height = self.train_parameters['height']
        self.channels = self.train_parameters['channels']

    def load_config(self):
        with open(self.config_name, 'r') as file:
            return json.load(file)
        
    def save_config(self):
        with open(self.config_name, 'w') as file:
            json.dump(self.data, file, indent=4)

    def set_epochs(self, epochs):
        self.train_parameters['epochs'] = epochs
        self.epochs = epochs

    def set_learning_rate(self, learning_rate):
        self.train_parameters['learningRate'] = learning_rate
        self.learning_rate = learning_rate

    def set_batch_size(self, batch_size):
        self.train_parameters['batchSize'] = batch_size
        self.batch_size = batch_size

    def set_width(self, width):
        self.train_parameters['width'] = width
        self.width = width

    def set_height(self, height):
        self.train_parameters['height'] = height
        self.height = height

    def set_channels(self, channels):
        self.train_parameters['channels'] = channels
        self.channels = channels

    def get_epochs(self):
        return self.epochs
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_channels(self):
        return self.channels

    def print_config(self):
        print("Train Parameters:")
        print(f"[1] Epochs: {self.epochs}")
        print(f"[2] Learning Rate: {self.learning_rate}")
        print(f"[3] Batch Size: {self.batch_size}")
        print(f"[4] Width x Height: {self.width}")
        print(f"[5] Channels: {self.channels}")

    def isInRange(self, value, min, max):
        if value >= min and value <= max:
            return True
        else:
            print(colored("Value must be between " + str(min) + " and " + str(max), 'red'))
            return False

    def get_parameters(self):
        return self.train_parameters
        