from functools import partial
from PIL import Image, ImageTk
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import Que, QMessage
import time
from tkinter import Tk, Label, Button, Frame, IntVar
from typing import List, Optional

from envy import DEBUG_MODE, TABLE_CASH_START, TABLE_CASH_SB, TABLE_CASH_BB, TBL_MOV, get_pos_names
from pologic.podeck import CRD_FIG, CRD_COL
from pologic.hand_history import HHistory, STATE
from podecide.stats.player_stats import PStatsEx

GUI_DELAY = 0.1 # seconds of delay for every message


# returns card graphic file name for given cards srt (e.g. 6D - six diamond)
def get_card_FN(
        imgs_FD,          # folder with gui images
        cs: str or None): # reverse for none
    if not cs: return f'{imgs_FD}/cards/dfR/REV0000.png'
    return            f'{imgs_FD}/cards/dfR/{cs}0000.png'

# builds tk images dict
def build_cards_img_dict(cards_FD):
    cD = {None: ImageTk.PhotoImage(Image.open(get_card_FN(cards_FD, None)))}
    for cf in CRD_FIG.values():
        if cf != 'X': # remove pad
            for cc in CRD_COL.values():
                cD[cf+cc] = ImageTk.PhotoImage(Image.open(get_card_FN(cards_FD, cf+cc)))
    return cD

# sets image of label
def set_image(lbl :Label, img :ImageTk.PhotoImage):
    lbl.configure(image=img)
    lbl.image = img


class GUI_HDMK:

    def __init__(
            self,
            players: List[str], # ids of players
            imgs_FD=    'gui/imgs',
            logger=     None,
            loglevel=   20,
    ):
        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger

        self.players = players
        self.logger.info(f'*** GUI_HDMK *** starts with players: {self.players}')

        self.tk_que = Que()
        self.out_que = Que()

        self.tk = Tk()
        self.tk.title('pypoks HDMK')
        self.tk.tk_setPalette(background='gray70')
        #self.tk.geometry('400x250+20+20')
        self.tk.resizable(False,False)
        self.tk.protocol("WM_DELETE_WINDOW", self.__on_closing)

        ico = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/aiico.png'))
        self.tk.iconphoto(False, ico)

        self.cards_imagesD = build_cards_img_dict(imgs_FD)
        self.tcards = [] # here hand table cards are stored
        self.tcsh_tc = 0 # will keep it to capture fold event
        self.pl_won = [0 for _ in range(len(self.players))]
        self.n_hands = 0
        self.players_cards = {ix:[] for ix in range(len(self.players))}

        self.states = [] # current hand states cache
        self.human_stats = PStatsEx(
            player=         0,
            use_initial=    False,
            upd_freq=       1,
            logger=         self.logger)

        pyp_lbl = Label(self.tk)
        pyp_lbl.grid(row=0, column=0)
        set_image(pyp_lbl, ImageTk.PhotoImage(Image.open(f'{imgs_FD}/pypoks_bar.png')))

        ### players frame ******************************************************************************** players frame

        self.__btn_pos = get_pos_names().index('BTN') # index of BTN position (for given N_TABLE_PLAYERS)

        pl_frm = Frame(self.tk, padx=5, pady=5)
        pl_frm.grid(row=1, column=0)
        self.plx_elD = {}
        self.dealer_img = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/dealer.png'))
        self.nodealer_img = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/no_dealer.png'))
        user_ico = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/user.png'))
        ai_ico = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/ai.png'))
        for ix in range(len(self.players)):
            plx_frm = Frame(pl_frm, padx=5, pady=5)
            plx_frm.grid(row=0, column=ix)
            plx_lblL = []
            lbl = Label(plx_frm, text=0, font=('Helvetica bold', 9), width=5, pady=2)  # won
            lbl.grid(row=0, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, bg='gray80') # icon
            lbl.grid(row=1, column=0)
            plx_lblL.append(lbl)
            set_image(lbl, ai_ico if ix else user_ico)
            lbl = Label(plx_frm, bg='gray80')  # dealer
            lbl.grid(row=2, column=0)
            plx_lblL.append(lbl)
            set_image(lbl, self.nodealer_img)
            lbl = Label(plx_frm, text=f'{ix}:{players[ix][:9]}', font=('Helvetica bold', 9), width=11, pady=1) # player name
            lbl.grid(row=3, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, font=('Helvetica bold', 12), width=5)
            lbl.grid(row=4, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, font=('Helvetica', 18), width=5)
            lbl.grid(row=5, column=0)
            plx_lblL.append(lbl)
            self.plx_elD[ix] = {'lblL': plx_lblL}

            self.__upd_plcsh(ix)

        ### table frame ************************************************************************************ table frame

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

        ### my frame ****************************************************************************************** my frame

        m_frm = Frame(self.tk, padx=5, pady=5)
        m_frm.grid(row=3, column=0)

        ### my cards subframe ************************************************************************ my cards subframe

        myc_frm = Frame(m_frm, padx=5, pady=5)
        myc_frm.grid(row=0, column=0)
        self.myc_lblL = []
        for ix in range(2):
            clbl = Label(myc_frm, pady=2, padx=2)
            clbl.grid(row=0, column=ix)
            self.myc_lblL.append(clbl)
        self.__upd_myc()

        ### decision subframe ************************************************************************ decision subframe

        mvnL = [mv[0] for mv in TBL_MOV] # moves names
        lcolL = []                       # fg colors in a frame
        for mvn in mvnL:
            if mvn == 'CCK':
                lcolL.append('dark green')
            if mvn == 'FLD':
                lcolL.append('black')
            if mvn == 'CLL':
                lcolL.append('DodgerBlue3')
            if 'BR' in mvn:
                lcolL.append('red')

        dec_frm = Frame(m_frm, padx=5, pady=5)
        dec_frm.grid(row=0, column=1)

        self.dec_lblL = []
        for ix in range(len(lcolL)):
            lbl = Label(dec_frm, fg=lcolL[ix], font=('Helvetica', 14))
            lbl.grid(row=0, column=ix)
            self.dec_lblL.append(lbl)
        self.__set_dec_lbl_val()

        self.dec_btnL = []
        for ix in range(len(mvnL)):
            btn = Button(dec_frm, text=mvnL[ix], fg=lcolL[ix], font=('Helvetica', 12), command=partial(self.__put_decision, ix), pady=2, padx=2, width=4)
            btn.grid(row=1,column=ix)
            self.dec_btnL.append(btn)
        self.__set_dec_btn_act()

        # GO
        go_frm = Frame(self.tk, padx=5, pady=5)
        go_frm.grid(row=4, column=0)
        self.next_go = IntVar()
        self.next_btn = Button(go_frm, text='GO', command=lambda: self.next_go.set(1), pady=2, padx=2, width=15)
        self.next_btn.grid(row=0, column=0, pady=5)
        self.next_btn['state'] = 'disabled'
        self.nHlbl = Label(go_frm, text=0, font=('Helvetica bold', 11), width=5)  # n_hands
        self.nHlbl.grid(row=0, column=1)

    ### GUI main logic methods ****************************************************************** GUI main logic methods

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
        while True:
            message = self.tk_que.get(block=False)
            #print(message)
            if not message: break
            if message.type == 'possible_moves':
                data = message.data
                cv = [data['moves_cash'][ix] if data['possible_moves'][ix] else '-' for ix in range(len(TBL_MOV))]
                self.__set_dec_lbl_val(cv)
                self.__set_dec_btn_act(data['possible_moves'])
            if message.type == 'state':
                self.__proc_state(message.data)
        self.__afterloop()

    # processes incoming state
    def __proc_state(self, state:STATE):

        self.states.append(state)

        prn = True # to catch unhandled states below

        prn_event = HHistory.readable_event(state)
        if prn_event and state[0] != 'PLH':
            print(prn_event)

        if state[0] == 'HST':
            self.n_hands += 1
            self.nHlbl['text'] = self.n_hands
            prn = False

        if state[0] in ['PSB', 'PBB']:
            prn = False

        if state[0] == 'TST':
            if state[1][0] == 0: # idle
                self.__upd_myc()
                self.__upd_tblc()
                self.__upd_tcsh()
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, TABLE_CASH_START)
                    self.__set_pl_active(plix)
            self.tcsh_tc = 0
            if state[1][0] != 1: # not preflop
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, True, None)
            prn = False

        # TODO: update here with player starting cash given with POS
        if state[0] == 'POS':
            # SB
            if state[1][1] == 0:
                self.__upd_plcsh(state[1][0], TABLE_CASH_START - TABLE_CASH_SB, TABLE_CASH_SB)
            # BB
            if state[1][1] == 1:
                self.__upd_plcsh(state[1][0], TABLE_CASH_START - TABLE_CASH_BB, TABLE_CASH_BB)
            # is BTN
            if state[1][1] == self.__btn_pos:
                self.__set_button(state[1][0])
            prn = False

        if state[0] == 'PLH':
            if state[1][0] == 0:
                self.__upd_myc(state[1][1], state[1][2])
            self.players_cards[state[1][0]] = state[1][1:]
            prn = False

        if state[0] == 'TCD':
            self.__upd_tblc(list(state[1]))
            prn = False

        if state[0] == 'T$$':
            self.__upd_tcsh(state[1][0] - state[1][1], state[1][1])
            self.tcsh_tc = state[1][2]
            prn = False

        if state[0] == 'MOV':
            # FLD case
            if TBL_MOV[state[1][1]][0] == 'FLD':
                self.__upd_plcsh(state[1][0], state[1][3][0]) # sets cash_cr to '-'
                self.__set_pl_active(state[1][0], False)
            else:
                self.__upd_plcsh(state[1][0], state[1][3][0] - state[1][2], state[1][3][2] + state[1][2])
            prn = False

        if state[0] == 'PRS':
            self.__upd_pl_won(state[1][0], state[1][1])
            prn = False

        if state[0] == 'HFN':
            if DEBUG_MODE:
                for ix in self.players_cards:
                    print(f'DEB: pl{ix} cards: {self.players_cards[ix][0]} {self.players_cards[ix][1]}')

            hh = HHistory()
            hh.events = self.states
            hh.save(file=f'hh_test_stats/human_{self.n_hands:02}.hh')
            self.logger.debug(str(hh))

            self.human_stats.process_states(self.states)
            self.logger.debug(f'human_stats: {self.human_stats.player_stats}')
            self.states = []

            self.next_btn['state'] = 'normal'
            print('\npress GO to start next hand')
            self.next_btn.wait_variable(self.next_go)
            self.next_btn['state'] = 'disabled'
            prn = False

        #prn = True
        self.tk.update_idletasks()
        if prn: print(f' >>> {state}')
        time.sleep(GUI_DELAY)

    # returns decision (decision button pressed)
    def __put_decision(self, dec:int):
        message = QMessage(type='decision', data=dec)
        self.out_que.put(message)
        self.__set_dec_lbl_val()
        self.__set_dec_btn_act()

    ### players frames methods ****************************************************************** players frames methods

    # updates player tot won
    def __upd_pl_won(
            self,
            plix: int,
            won):
        self.pl_won[plix] += int(won)
        self.plx_elD[plix]['lblL'][0]['text'] = self.pl_won[plix]

    # updates player cash
    def __upd_plcsh(
            self,
            plix: int,
            csh: int=       None,   # True does not update
            csh_cr: int=    None):  # True does not update
        if csh is None: csh = '-'
        if csh_cr is None: csh_cr = '-'
        if csh is not True:     self.plx_elD[plix]['lblL'][4]['text'] = csh
        if csh_cr is not True:  self.plx_elD[plix]['lblL'][5]['text'] = csh_cr

    def __set_pl_active(self, plix:int, a=True):
        self.plx_elD[plix]['lblL'][4]['fg'] = 'black' if a else 'gray36'
        self.plx_elD[plix]['lblL'][5]['fg'] = 'black' if a else 'gray36'

    def __set_button(self, i:int=None):
        set_image(self.plx_elD[i]['lblL'][2], self.dealer_img)
        other = list(range(len(self.players)))
        other.pop(i)
        for ix in other:
            set_image(self.plx_elD[ix]['lblL'][2], self.nodealer_img)

    ### table frame methods ************************************************************************ table frame methods

    # updates self.tcards list
    def __upd_tblc(self, cl:Optional[List]=None):

        # update list
        if not cl:
            self.tcards = []
        else:
            self.tcards += cl

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

    ### my cards frame methods ****************************************************************** my cards frame methods

    # updates my cards
    def __upd_myc(self, ca :str=None, cb :str=None):
        set_image(self.myc_lblL[0], self.cards_imagesD[ca])
        set_image(self.myc_lblL[1], self.cards_imagesD[cb])

    ### decision frame methods ****************************************************************** decision frame methods

    # sets $ values of labels
    def __set_dec_lbl_val(self, val:Optional[List[int]]=None):
        if not val: val = ['-']*len(TBL_MOV)
        for ix in range(len(self.dec_lblL)):
            self.dec_lblL[ix]['text'] = val[ix]

    # sets state of buttons
    def __set_dec_btn_act(self, act:Optional[List[bool]]=None):
        if not act:
            act = [False]*len(TBL_MOV)
        for ix in range(len(self.dec_btnL)):
            self.dec_btnL[ix]['state'] = 'normal' if act[ix] else 'disabled'

    def __on_closing(self):
        self.next_btn.invoke() # this button may hold exit (#286), we need to invoke it
        self.tk.quit()