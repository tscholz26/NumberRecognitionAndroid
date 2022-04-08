#local libraries
from network import Network

#third-party libraries
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class NumberRecognition(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        #self.window.rows = 1

        #widgets hinzufügen
        self.window.add_widget(Image(source="data/bild.jpg"))
        
        self.textdings = Label(text = "texttest")
        self.window.add_widget(self.textdings)

        self.button = Button(text="Ausfuehren")
        self.button.bind(on_press=self.calc)
        self.window.add_widget(self.button)

        return self.window

    def calc(self, event):
        self.textdings.text = "calced"


if __name__ == "__main__":
    NumberRecognition().run()


    
#gabi = Network([],"best/god_784$300§60$10_99k64")

