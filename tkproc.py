"""

 2020 (c) piteren

"""

from multiprocessing import Queue
from tkinter import Tk, Label, Button



class TkProc:

    def __init__(self):
        self.tk = Tk()
        self.tk_que = Queue()
        self.out_que = Queue()
        self.runh = None
        self.message_event = '<<message>>'
        self.tk.bind(self.message_event, self.process_message_queue)

    def run_tk(self):
        self.tk.title('My Window')  # add buttons, etc.

        label = Label(self.tk, text="This is our first GUI!")
        label.pack()

        greet_button = Button(self.tk, text="Greet", command=self.runh)
        greet_button.pack()

        close_button = Button(self.tk, text="Close", command=self.tk.quit)
        close_button.pack()

        self.tk.lift()
        self.tk.mainloop()

    def send_message_to_ui(self, message):
        self.tk_que.put(message)
        self.tk.event_generate(self.message_event, when='tail')

    def process_message_queue(self, event):
        #print('@@@ event',event) # ?
        #while not self.tk_que.empty():
        message = self.tk_que.get()
        label = Label(self.tk, text=str(message))
        label.pack()
        #print(f' # tk received: {message}')
        # process the message here

    def greet(self):
        print("Greetings!")