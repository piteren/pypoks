"""

 2020 (c) piteren

"""

from functools import partial
from multiprocessing import Queue
import time
from tkinter import Tk, Label, Button, Frame, IntVar
from PIL import Image, ImageTk

from pologic.poenvy import TBL_MOV, N_TABLE_PLAYERS, TABLE_CASH_START, TABLE_SB, TABLE_BB, DEBUG_MODE
from pologic.podeck import CRD_FIG, CRD_COL

# returns card graphic file name for given cards srt (e.g. 6D - six diamond)
def get_card_FN(cs: str or None):
    spath = 'gui/cards/dfR/'
    epath = '0000.png'
    if not cs: return spath + 'REV' + epath
    return spath + cs + epath

# builds tk images dict
def build_cards_img_dict():
    cD = {None: ImageTk.PhotoImage(Image.open(get_card_FN(None)))}
    for cf in CRD_FIG.values():
        if cf != 'X': # remove pad
            for cc in CRD_COL.values():
                cD[cf+cc] = ImageTk.PhotoImage(Image.open(get_card_FN(cf+cc)))
    return cD

# sets image of label
def set_image(lbl :Label, img :ImageTk.PhotoImage):
    lbl.configure(image=img)
    lbl.image = img


class TkProc:

    def __init__(
            self,
            n_players=  N_TABLE_PLAYERS):

        self.tk_que = Queue()
        self.out_que = Queue()

        self.tk = Tk()
        self.tk.title('pypoks HDMK')
        self.tk.tk_setPalette(background='gray70')
        #self.tk.geometry('400x250+20+20')
        self.tk.resizable(0,0)

        self.cards_imagesD = build_cards_img_dict()
        self.tcards = [] # here hand table cards are stored
        self.tcsh_tc = 0 # will keep it for fold capture
        self.ops_cards = {1:[],2:[]}

        pyp_lbl = Label(self.tk)
        pyp_lbl.grid(row=0, column=0)
        set_image(pyp_lbl, ImageTk.PhotoImage(Image.open('gui/pypoks_bar.png')))


        # players frame ************************************************************************************************

        pl_frm = Frame(self.tk, padx=5, pady=5)
        pl_frm.grid(row=1, column=0)
        self.plx_elD = {}
        self.dealer_img = ImageTk.PhotoImage(Image.open('gui/dealer.png'))
        self.nodealer_img = ImageTk.PhotoImage(Image.open('gui/no_dealer.png'))
        user_ico = ImageTk.PhotoImage(Image.open('gui/user.png'))
        ai_ico = ImageTk.PhotoImage(Image.open('gui/ai.png'))
        for ix in range(n_players):
            plx_frm = Frame(pl_frm, padx=5, pady=5)
            plx_frm.grid(row=0, column=ix)

            plx_lblL = []
            lbl = Label(plx_frm) # icon
            lbl.grid(row=0, column=0)
            plx_lblL.append(lbl)
            set_image(lbl, ai_ico if ix else user_ico)
            lbl = Label(plx_frm)  # dealer
            lbl.grid(row=1, column=0)
            plx_lblL.append(lbl)
            set_image(lbl, self.nodealer_img)
            lbl = Label(plx_frm, font=('Helvetica bold', 12), width=5)
            lbl.grid(row=2, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, font=('Helvetica', 18), width=5)
            lbl.grid(row=3, column=0)
            plx_lblL.append(lbl)
            self.plx_elD[ix] = {'lblL': plx_lblL}

            self.__upd_plcsh(ix)


        # table frame **************************************************************************************************

        tbl_frm = Frame(self.tk, padx=5, pady=5)
        tbl_frm.grid(row=2, column=0)

        tcrds_frm = Frame(tbl_frm, padx=5, pady=5)
        tcrds_frm.grid(row=0, column=0)
        self.tblc_lblL = []
        for ix in range(5):
            clbl = Label(tcrds_frm, pady=2, padx=2)
            clbl.grid(row=0, column=ix)
            self.tblc_lblL.append(clbl)
        self.__upd_tblc()

        tcsh_frm = Frame(tbl_frm, padx=5, pady=5)
        tcsh_frm.grid(row=0, column=1)
        self.tcsh_lblL = []
        lbl = Label(tcsh_frm, font=('Helvetica bold', 18), width=5)
        lbl.grid(row=0, column=0)
        self.tcsh_lblL.append(lbl)
        lbl = Label(tcsh_frm, font=('Helvetica', 12), width=5)
        lbl.grid(row=1, column=0)
        self.tcsh_lblL.append(lbl)
        self.__upd_tcsh()

        # my frame *****************************************************************************************************

        m_frm = Frame(self.tk, padx=5, pady=5)
        m_frm.grid(row=3, column=0)

        # my cards frame ***********************************************************************************************

        myc_frm = Frame(m_frm, padx=5, pady=5)
        myc_frm.grid(row=0, column=0)
        self.myc_lblL = []
        for ix in range(2):
            clbl = Label(myc_frm, pady=2, padx=2)
            clbl.grid(row=0, column=ix)
            self.myc_lblL.append(clbl)
        self.__upd_myc()

        # decision frame ***********************************************************************************************

        lcol = ['black', 'DodgerBlue3'] + ['red'] * (len(TBL_MOV) - 2)  # fg colors in frame
        mnm = [TBL_MOV[k] for k in sorted(list(TBL_MOV.keys()))]  # moves names
        dec_frm = Frame(m_frm, padx=5, pady=5)
        dec_frm.grid(row=0, column=1)

        self.dec_lblL = []
        for ix in range(len(lcol)):
            lbl = Label(dec_frm, fg=lcol[ix], font=('Helvetica', 14))
            lbl.grid(row=0, column=ix)
            self.dec_lblL.append(lbl)
        self.__set_dec_lbl_val(['-','-','-','-'])

        self.dec_btnL = []
        for ix in range(len(mnm)):
            btn = Button(dec_frm, text=mnm[ix], fg=lcol[ix], font=('Helvetica', 12), command=partial(self.__put_decision, ix), pady=2, padx=2, width=4)
            btn.grid(row=1,column=ix)
            self.dec_btnL.append(btn)
        self.__set_dec_btn_act([0,0,0,0])

        # next
        self.next_go = IntVar()
        self.next_btn = Button(self.tk, text='GO', command=lambda: self.next_go.set(1), pady=2, padx=2, width=15)
        self.next_btn.grid(row=4, column=0, pady=5)
        self.next_btn['state'] = 'disabled'

    # GUI main logic methods ******************************************************************************** main logic

    # runs main loop
    def run_tk(self):
        self.tk.lift()
        self.__afterloop()
        self.tk.mainloop()

    # after
    def __afterloop(self, ms :int=500):
        self.tk.after(ms, self.__check_message_queue)

    # checks input que
    def __check_message_queue(self):
        while not self.tk_que.empty():
            message = self.tk_que.get()
            if type(message) is list:
                self.__proc_message(message)
            if type(message) is dict:
                cv = [message['moves_cash'][ix] if message['possible_moves'][ix] else '-' for ix in range(4)]
                self.__set_dec_lbl_val(cv)
                self.__set_dec_btn_act(message['possible_moves'])
        self.__afterloop()

    # processes incomming messages
    def __proc_message(self, message):
        prn = True

        if message[0] == 'HST':
            if DEBUG_MODE:
                if self.ops_cards[1]:
                    for ix in [1,2]:
                        print(f' DEBUG: player {ix} cards: {self.ops_cards[ix][0]} {self.ops_cards[ix][1]}')
            self.next_btn['state'] = 'normal'
            print('\npress GO to start next hand')
            self.next_btn.wait_variable(self.next_go)
            self.next_btn['state'] = 'disabled'
            prn = False

        if message[0] in ['PSB', 'PBB']: prn = False

        if message[0] == 'TST':
            if message[1] == 'idle':
                print('\n ***** hand starts')
                self.__upd_myc()
                self.__upd_tblc()
                self.__upd_tcsh()
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, TABLE_CASH_START)
                    self.__set_pl_active(plix)
            else: print(f' * {message[1]}')
            self.tcsh_tc = 0
            if message[1] != 'preflop':
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, True, None)
            prn = False

        if message[0] == 'POS':
            if message[1][1] == 'SB': self.__upd_plcsh(message[1][0], TABLE_CASH_START-TABLE_SB, TABLE_SB)
            if message[1][1] == 'BB': self.__upd_plcsh(message[1][0], TABLE_CASH_START-TABLE_BB, TABLE_BB)
            if message[1][1] == 'BTN': self.__set_button(message[1][0])
            prn = False

        if message[0] == 'PLH':
            if message[1][0] == 0:
                self.__upd_myc(message[1][1], message[1][2])
            else: self.ops_cards[message[1][0]] = message[1][1:]
            prn = False

        if message[0] == 'TCD':
            self.__upd_tblc(message[1])
            prn = False

        if message[0] == 'T$$':
            self.__upd_tcsh(message[1][0]-message[1][1], message[1][1])
            self.tcsh_tc = message[1][2]
            prn = False

        if message[0] == 'MOV':
            # fold case
            if message[1][1] == 'C/F' and message[1][2] < self.tcsh_tc - message[1][3][2]:
                self.__upd_plcsh(message[1][0], message[1][3][0])
                self.__set_pl_active(message[1][0], False)
            else:
                self.__upd_plcsh(message[1][0], message[1][3][0]-message[1][2], message[1][3][2]+message[1][2])
            prn = False

        if message[0] == 'PRS':
            r = message[1][2] if type(message[1][2]) is str else message[1][2][-1]
            print(f' $$$: player {message[1][0]} {message[1][1]} {r}')
            prn = False

        self.tk.update_idletasks()
        if prn: print(f' >>> {message}')
        time.sleep(0.2)

    # returns decision (decision button pressed)
    def __put_decision(self, dec :int):
        self.out_que.put(dec)
        self.__set_dec_btn_act([0,0,0,0])

    # decision frame methods **************************************************************************** decision frame

    # sets $ values oflabels
    def __set_dec_lbl_val(self, val :list):
        for ix in range(len(self.dec_lblL)):
            self.dec_lblL[ix]['text'] = val[ix]

    # sets state of buttons
    def __set_dec_btn_act(self, act :list):
        for ix in range(len(self.dec_btnL)):
            self.dec_btnL[ix]['state'] = 'normal' if act[ix] else'disabled'

    # my cards frame methods **************************************************************************** my cards frame

    # updates my cards
    def __upd_myc(self, ca :str=None, cb :str=None):
        set_image(self.myc_lblL[0], self.cards_imagesD[ca])
        set_image(self.myc_lblL[1], self.cards_imagesD[cb])

    # table frame methods ********************************************************************************** table frame

    # updates self.tcards list
    def __upd_tblc(self, cl :list=None):

        # update list
        if not cl: self.tcards = []
        else: self.tcards += cl

        # update GUI
        cl = [] + self.tcards # copy (!)
        cl += [None]*(5-len(cl))
        for ix in range(5):
            set_image(self.tblc_lblL[ix], self.cards_imagesD[cl[ix]])

    # updates table cash
    def __upd_tcsh(self, a :int=None, b :int=None):
        if a is None: a = '-'
        if b is None: b = '-'
        self.tcsh_lblL[0]['text'] = a
        self.tcsh_lblL[1]['text'] = f'({b})'

    # players frames ************************************************************************************ players frames

    # updates player cash
    def __upd_plcsh(
            self,
            plix :int,
            csh :int=       None,   # True does not update
            csh_cr :int=    None):  # True does not update
        if csh is None: csh = '-'
        if csh_cr is None: csh_cr = '-'
        if csh is not True:     self.plx_elD[plix]['lblL'][2]['text'] = csh
        if csh_cr is not True:  self.plx_elD[plix]['lblL'][3]['text'] = csh_cr

    def __set_pl_active(self, plix :int, a=True):
        self.plx_elD[plix]['lblL'][2]['fg'] = 'black' if a else 'gray36'
        self.plx_elD[plix]['lblL'][3]['fg'] = 'black' if a else 'gray36'

    def __set_button(self, i :int=None):
        set_image(self.plx_elD[i]['lblL'][1], self.dealer_img)
        other = [0, 1, 2]
        other.pop(i)
        for ix in other:
            set_image(self.plx_elD[ix]['lblL'][1], self.nodealer_img)

    # other ****************************************************************************************************** other
