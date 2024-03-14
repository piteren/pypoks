"""

 2020 (c) piteren

 multiprocessing and tkinter:
    https://www.reddit.com/r/learnpython/comments/6flxe7/multiprocessingthreads_and_tkinter/
    https://stackoverflow.com/questions/16745507/tkinter-how-to-use-threads-to-preventing-main-event-loop-from-freezing

"""

from tkinter import Tk, Label, Button

class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        print('initialized!')

    def greet(self):
        print("Greetings!")


if __name__ == "__main__":

    root = Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()
    print('finished')