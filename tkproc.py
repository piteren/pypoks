"""

 2020 (c) piteren

"""

from functools import partial
from multiprocessing import Queue
from tkinter import Tk, Label, Button, Frame


class TkProc:

    def __init__(self):
        self.tk = Tk()
        self.tk_que = Queue()
        self.out_que = Queue()

        self.tk.title('pypoks HDMK')
        self.tk.geometry('400x250+100+100')
        self.tk.resizable(0,0)

        self.cards = []
        self.table_cash = 0
        self.dec_btnL = []

        self.dec_btnL.append(Button(self.tk, text='C/F', command=partial(self.put_decision,0), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.tk, text='CLL', command=partial(self.put_decision,1), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.tk, text='BR5', command=partial(self.put_decision,2), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.tk, text='BR8', command=partial(self.put_decision,3), pady=2, padx=2, width=4))
        for ix in range(len(self.dec_btnL)): self.dec_btnL[ix].grid(row=0,column=ix)
        self.__set_dec_btn_act([0,0,0,0])

        #self.tk.grid_columnconfigure(0, weight=1)
        #self.tk.grid_columnconfigure(5, weight=1)
        subframe = Frame(self.tk)
        buttonA = Button(subframe).grid(row=0,column=0)
        buttonB = Button(subframe).grid(row=0,column=1)
        subframe.grid(row=1,column=4)

    def run_tk(self):
        self.tk.lift()
        self.__afterloop()
        self.tk.mainloop()

    def __afterloop(self, ms :int=500):
        self.tk.after(ms, self.check_message_queue)

    def __set_dec_btn_act(self, act :list):
        for ix in range(len(self.dec_btnL)):
            self.dec_btnL[ix]['state'] = 'normal' if act[ix] else'disabled'

    def __update_cards(self):
        pass

    def __print_state(self, state):
        if state[0] == 'TST':
            if state[1] == 'idle':  print('\n ***** hand starts')
            else:                   print(f' * {state[1]}')

    def check_message_queue(self):
        while not self.tk_que.empty():
            message = self.tk_que.get()
            print(message)
        self.__afterloop()

    def put_decision(self, dec :int):
        self.out_que.put(dec)
        self.__set_dec_btn_act([0,0,0,0])
