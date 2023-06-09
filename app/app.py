from components.mainMenu import MainMenu
from termcolor import colored

class App:
    def __init__(self):
        print(colored("""
================ WARNING ==================
Wersja GitHub nie posiada wyuczonego modelu!
===========================================""", 'red'))
        input("Naciśnij ENTER aby kontynuować...")
        self.mainMenu = MainMenu().run()
        
App()