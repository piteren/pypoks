"""

 2020 (c) piteren

"""

from functools import partial
from multiprocessing import Queue
from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk


def set_image(lbl :Label, img :ImageTk.PhotoImage):
    lbl.configure(image=img)
    lbl.image = img


class TkProc:

    def __init__(
            self,
            n_players):

        self.tk_que = Queue()
        self.out_que = Queue()

        self.tk = Tk()
        self.tk.title('pypoks HDMK')
        self.tk.tk_setPalette(background='gray70')
        #self.tk.geometry('400x250+20+20')
        self.tk.resizable(0,0)

        # decision frame ***********************************************************************************************
        self.dec_frm = Frame(self.tk, padx=5)
        self.dec_frm.grid(row=0,column=0)
        self.dec_lblL = []
        self.dec_lblL.append(Label(self.dec_frm, fg='black',       font=('Helvetica', 14)))
        self.dec_lblL.append(Label(self.dec_frm, fg='DodgerBlue3', font=('Helvetica', 14)))
        self.dec_lblL.append(Label(self.dec_frm, fg='orange red',  font=('Helvetica', 14)))
        self.dec_lblL.append(Label(self.dec_frm, fg='red',         font=('Helvetica', 14)))
        for ix in range(len(self.dec_lblL)): self.dec_lblL[ix].grid(row=0, column=ix)
        self.__set_dec_lbl_val([0,0,0,0])

        self.dec_btnL = []
        self.dec_btnL.append(Button(self.dec_frm, text='C/F', font=('Helvetica', 12), command=partial(self.put_decision,0), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.dec_frm, text='CLL', font=('Helvetica', 12), command=partial(self.put_decision,1), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.dec_frm, text='BR5', font=('Helvetica', 12), command=partial(self.put_decision,2), pady=2, padx=2, width=4))
        self.dec_btnL.append(Button(self.dec_frm, text='BR8', font=('Helvetica', 12), command=partial(self.put_decision,3), pady=2, padx=2, width=4))
        for ix in range(len(self.dec_btnL)): self.dec_btnL[ix].grid(row=1,column=ix)
        self.__set_dec_btn_act([0,0,0,0])


        self.imgA = ImageTk.PhotoImage(Image.open('gui/cards/dfR/9H0000.png'))
        self.imgB = ImageTk.PhotoImage(Image.open('gui/cards/dfR/REV0000.png'))
        # my cards frame ***********************************************************************************************
        self.myc_frm = Frame(self.tk, padx=5)
        self.myc_frm.grid(row=0,column=1)
        self.myc_lblL = []
        self.myc_lblL.append(Label(self.myc_frm, pady=2, padx=2))
        self.myc_lblL.append(Label(self.myc_frm, pady=2, padx=2))
        for ix in range(len(self.myc_lblL)): self.myc_lblL[ix].grid(row=2, column=ix)
        self.__set_myc(self.imgA, self.imgB)

        """
        self.tk.grid_columnconfigure(0, weight=1)
        self.tk.grid_columnconfigure(5, weight=1)
        
        subframe = Frame(self.tk)
        buttonA = Button(subframe).grid(row=0,column=0)
        buttonB = Button(subframe).grid(row=0,column=1)
        subframe.grid(row=1,column=4)
        """

    def run_tk(self):
        self.tk.lift()
        self.__afterloop()
        self.tk.mainloop()

    def __afterloop(self, ms :int=500):
        self.tk.after(ms, self.check_message_queue)

    def __set_dec_lbl_val(self, val :list):
        for ix in range(len(self.dec_lblL)):
            self.dec_lblL[ix]['text'] = val[ix]

    def __set_dec_btn_act(self, act :list):
        for ix in range(len(self.dec_btnL)):
            self.dec_btnL[ix]['state'] = 'normal' if act[ix] else'disabled'

    def __set_myc(self, ca, cb):
        set_image(self.myc_lblL[0], ca)
        set_image(self.myc_lblL[1], cb)


    def __update_cards(self):
        pass

    def __print_state(self, state):
        if state[0] == 'TST':
            if state[1] == 'idle':  print('\n ***** hand starts')
            else:                   print(f' * {state[1]}')

    def check_message_queue(self):
        while not self.tk_que.empty():
            message = self.tk_que.get()
            if type(message) is list: print(message)
            if type(message) is dict:
                self.__set_dec_lbl_val([message['moves_cash'][ix] for ix in range(4)])
                self.__set_dec_btn_act(message['possible_moves'])
        self.__afterloop()

    def put_decision(self, dec :int):
        self.out_que.put(dec)
        self.__set_dec_btn_act([0,0,0,0])
