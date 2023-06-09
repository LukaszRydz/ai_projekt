import json
from datetime import datetime
import os
import matplotlib.pyplot as plt

class Logs:
    folder_name = ""
    folder_path = ""
    json_file_path = ""

    jsonLog = None

    def __init__(self):
        # Create folder with actual data and time
        self.folder_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.folder_path = os.path.join(os.getcwd(), "logs", self.folder_name)

        # Create folder
        os.mkdir(self.folder_path)

        # Create log file path
        self.json_file_path = os.path.join(self.folder_path, "log.json")
        
        self.jsonLog = {}

    def get_folder_path(self):
        return self.folder_path
    
    def get_json_file_path(self):
        return self.json_file_path
    
    def add_value(self, key, value):
        if os.path.isfile(self.json_file_path):
            with open(self.json_file_path, 'r') as file:
                self.jsonLog = json.load(file)

        self.jsonLog[key] = value

        # Zapisywanie słownika do pliku .json
        with open(self.json_file_path, 'w') as file:
            json.dump(self.jsonLog, file)

    def savePlot(self, TRAINING_HISTORY, trainParameters):
        plt.style.use("seaborn")
        plt.figure(figsize=(10, 6))
        epochs = range(1, trainParameters.epochs + 1)
        
        plt.plot(epochs, TRAINING_HISTORY.history["loss"], label="Training Loss", color="blue", linestyle="-")
        plt.plot(epochs, TRAINING_HISTORY.history["val_loss"], label="Validation Loss", color="red", linestyle="--")
        plt.plot(epochs, TRAINING_HISTORY.history["acc"], label="Training Accuracy", color="green", linestyle="-")
        plt.plot(epochs, TRAINING_HISTORY.history["val_acc"], label="Validation Accuracy", color="orange", linestyle="--")

        # Zaznacz punkty szczytowe dla val_acc
        val_acc = TRAINING_HISTORY.history["val_acc"]
        max_val_acc = max(val_acc)
        max_val_acc_index = val_acc.index(max_val_acc)
        plt.scatter(max_val_acc_index + 1, max_val_acc, color='red', label=f"Peak: {max_val_acc:.4f}")
        plt.text(max_val_acc_index + 1, max_val_acc, f"({max_val_acc_index + 1}, {max_val_acc:.4f})", ha='right', va='bottom')

        plt.title("Model Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower right")

        # Zapisz wykres do folderu
        plot_path = os.path.join(self.folder_path, "plot.png")
        plt.savefig(plot_path)
        plt.close()