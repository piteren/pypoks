from functools import partial
from PIL import Image, ImageTk
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.mpython.mptools import Que, QMessage
import time
from tkinter import Tk, Label, Button, Frame, IntVar
from typing import List, Optional

from envy import get_pos_names
from pologic.game_config import GameConfig
from pologic.podeck import CRD_FIG, CRD_COL
from pologic.hand_history import STATE
from podecide.stats.player_stats import PStatsEx

GUI_DELAY = 0.1 # seconds of delay for every message


def get_card_FN(
        imgs_FD,          # folder with gui images
        cs: Optional[str] # card str (e.g. 6D - six diamond), None gives reverse
):
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


class HumanGameGUI:

    def __init__(
            self,
            players: List[str],         # ids of players
            game_config: GameConfig,
            imgs_FD=        'gui/imgs',
            logger=         None,
            loglevel=       20,
    ):
        if not logger:
            logger = get_pylogger(name=self.__class__.__name__, level=loglevel)
        self.logger = logger

        self.players = players
        self.logger.info(f'*** {self.__class__.__name__} *** starts with players: {self.players}')

        self.gc = game_config

        # ques to communicate with DMK
        self.queI = Que()
        self.queO = Que()

        self.que_to_gm = Que() # here GUI sends next_hand / exit decision to GM

        self.tk = Tk()
        self.tk.title('pypoks HumanGame')
        self.tk.tk_setPalette(background='gray80')
        self.tk.resizable(False,False)
        self.tk.protocol("WM_DELETE_WINDOW", self.close_window)

        ico = ImageTk.PhotoImage(Image.open(f'{imgs_FD}/aiico.png'))
        self.tk.iconphoto(False, ico)

        self.cards_imagesD = build_cards_img_dict(imgs_FD)
        self.tcards = [] # here hand table cards are saved
        self.pl_won = [0 for _ in range(len(self.players))]
        self.n_hands = 0
        self.players_cards = {ix:[] for ix in range(len(self.players))}
        self.hand_is_finished = True

        self.states = [] # current hand states cache
        self.human_stats = PStatsEx(
            player=         0,
            table_size=     self.gc.table_size,
            table_moves=    self.gc.table_moves,
            use_initial=    False,
            upd_freq=       1,
            logger=         self.logger)

        pyp_lbl = Label(self.tk)
        pyp_lbl.grid(row=0, column=0)
        set_image(pyp_lbl, ImageTk.PhotoImage(Image.open(f'{imgs_FD}/pypoks_bar_alpha_white.png')))

        ### players frame ******************************************************************************** players frame

        self.__btn_pos = get_pos_names(self.gc.table_size).index('BTN') # index of BTN position (for given table_size)

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
            lbl = Label(plx_frm, text=f'{ix}:{self.players[ix][:9]}', font=('Helvetica bold', 9), width=11, pady=1) # player name
            lbl.grid(row=3, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, font=('Helvetica bold', 18), width=6)
            lbl.grid(row=4, column=0)
            plx_lblL.append(lbl)
            lbl = Label(plx_frm, font=('Helvetica', 12), width=6)
            lbl.grid(row=5, column=0)
            plx_lblL.append(lbl)
            self.plx_elD[ix] = {'lblL': plx_lblL}

            self.__upd_plcsh(ix)

        ### table frame ************************************************************************************ table frame

        tbl_frm = Frame(self.tk, padx=5, pady=5)
        tbl_frm.grid(row=2, column=0)

        # my cards subframe
        myc_frm = Frame(tbl_frm, padx=15, pady=5)
        myc_frm.grid(row=0, column=0)
        self.myc_lblL = []
        for ix in range(2):
            clbl = Label(myc_frm, pady=2, padx=2)
            clbl.grid(row=0, column=ix)
            self.myc_lblL.append(clbl)
        self.__upd_myc()

        # table cards
        tcrds_frm = Frame(tbl_frm, padx=5, pady=5)
        tcrds_frm.grid(row=0, column=1)
        self.tblc_lblL = []
        for ix in range(5):
            clbl = Label(tcrds_frm, pady=2, padx=2)
            clbl.grid(row=0, column=ix)
            self.tblc_lblL.append(clbl)
        self.__upd_tblc()

        # table cash
        tcsh_frm = Frame(tbl_frm, padx=5, pady=5)
        tcsh_frm.grid(row=0, column=2)
        self.tcsh_lblD = {
            'street':    Label(tcsh_frm, font=('Helvetica', 12), fg='red4', width=10),
            'pot':      Label(tcsh_frm, font=('Helvetica bold', 18), width=10)}
        self.tcsh_lblD['street'].grid(row=0, column=0)
        self.tcsh_lblD['pot'].grid(row=1, column=0)
        self.__upd_tcash()

        ### decision frame ****************************************************************************** decision frame

        dec_frm = Frame(self.tk, padx=5, pady=5)
        dec_frm.grid(row=3, column=0)

        move_names = [f'{mov[0]}' for mov in self.gc.table_moves]
        self.move_math = {
            1: ['' if len(mov) == 1 else f'{mov[1]}x' for mov in self.gc.table_moves],
            2: ['' if len(mov) == 1 else f'{int(mov[2]*100)}%' for mov in self.gc.table_moves]}

        # prepare fg colors in a frame
        lcolL = []
        for mvn in move_names:
            if mvn == 'CCK': lcolL.append('dark green')
            if mvn == 'FLD': lcolL.append('black')
            if mvn == 'CLL': lcolL.append('DodgerBlue3')
            if 'BR' in mvn:  lcolL.append('red')

        self.dec_mathL = []
        for ix in range(len(lcolL)):
            lbl = Label(dec_frm, fg=lcolL[ix], font=('Helvetica', 9))
            lbl.grid(row=0, column=ix)
            self.dec_mathL.append(lbl)
        self.__set_dec_math_text(1)

        self.dec_cashL = []
        for ix in range(len(lcolL)):
            lbl = Label(dec_frm, fg=lcolL[ix], font=('Helvetica', 14))
            lbl.grid(row=1, column=ix)
            self.dec_cashL.append(lbl)
        self.__set_dec_cash_val()

        self.dec_btnL = []
        for ix in range(len(move_names)):
            btn = Button(dec_frm, text=move_names[ix], fg=lcolL[ix], font=('Helvetica',11), command=partial(self.__put_decision, ix), pady=5, padx=5, width=4)
            btn.grid(row=2,column=ix)
            self.dec_btnL.append(btn)
        self.__set_dec_btn_act()

        # next hand / exit
        go_frm = Frame(self.tk, padx=5, pady=5)
        go_frm.grid(row=4, column=0)
        self.next_go = IntVar()
        self.next_btn = Button(go_frm, text='next hand', command=lambda: self.next_go.set(1), pady=2, padx=2, width=15)
        self.next_btn.grid(row=0, column=0, pady=5)
        self.next_btn['state'] = 'disabled'
        self.nHlbl = Label(go_frm, text=0, font=('Helvetica bold', 11), width=5)  # n_hands
        self.nHlbl.grid(row=0, column=1)
        self.exit_btn = Button(go_frm, text='exit', command=self.close_window, pady=2, padx=2, width=15)
        self.exit_btn.grid(row=0, column=2, pady=5)
        self.exit_btn['state'] = 'disabled'

    ### GUI main logic methods ****************************************************************** GUI main logic methods

    def close_window(self):
        if self.hand_is_finished:
            self.que_to_gm.put(QMessage(type='exit'))
            self.next_btn.invoke()  # this button may hold exit, needs to be invoked
            self.tk.quit()

    def run_loop(self):
        """ runs main loop """
        self.tk.lift()
        self.__afterloop()
        self.tk.mainloop()

    def __afterloop(self, ms:int=500):
        self.tk.after(ms, self.__check_message_queue)

    def __check_message_queue(self):
        """ checks input que """
        while True:
            message = self.queI.get(block=False)
            if not message:
                break
            if message.type == 'allowed_moves':
                data = message.data
                cv = [data['moves_cash'][ix] if data['allowed_moves'][ix] else '-' for ix in range(len(self.gc.table_moves))]
                self.__set_dec_cash_val(cv)
                self.__set_dec_btn_act(data['allowed_moves'])
            if message.type == 'state':
                self.__proc_state(message.data)
        self.__afterloop()

    def __proc_state(self, state:STATE):
        """ processes incoming state """

        self.states.append(state)

        prn = True # to catch unhandled states below

        if state[0] == 'GCF':
            prn = False

        if state[0] == 'HST':
            self.n_hands += 1
            self.nHlbl['text'] = self.n_hands
            self.hand_is_finished = False
            prn = False

        if state[0] in ['PSB', 'PBB']:
            prn = False

        if state[0] == 'TST':

            # idle
            if state[1][0] == 0:
                self.__upd_myc()
                self.__upd_tblc()
                self.__upd_tcash()
                self.__set_dec_math_text(1)
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, self.gc.table_cash_start)
                    self.__set_pl_active(plix)

            # postflop
            if state[1][0] == 2:
                self.__set_dec_math_text(2)

            if state[1][0] != 1: # not preflop
                for plix in self.plx_elD:
                    self.__upd_plcsh(plix, True, None)

            prn = False

        if state[0] == 'POS':
            # SB
            if state[1][1] == 0:
                self.__upd_plcsh(state[1][0], self.gc.table_cash_start - self.gc.table_cash_sb, self.gc.table_cash_sb)
            # BB
            if state[1][1] == 1:
                self.__upd_plcsh(state[1][0], self.gc.table_cash_start - self.gc.table_cash_bb, self.gc.table_cash_bb)
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
            self.__upd_tcash(street=state[1][1], pot=state[1][0])
            prn = False

        if state[0] == 'MOV':
            # FLD case
            if self.gc.table_moves[state[1][1]][0] == 'FLD':
                self.__upd_plcsh(state[1][0], state[1][4][0]) # sets cash_cs to '-'
                self.__set_pl_active(state[1][0], False)
            else:
                self.__upd_plcsh(state[1][0], state[1][4][0] - state[1][2], state[1][4][2] + state[1][2])
            prn = False

        if state[0] == 'PRS':
            self.__upd_pl_won(state[1][0], state[1][1])
            prn = False

        if state[0] == 'HFN':
            self.hand_is_finished = True
            self.next_btn['state'] = 'normal'
            self.exit_btn['state'] = 'normal'
            self.next_btn.wait_variable(self.next_go)
            message = QMessage(type='next_hand')
            self.que_to_gm.put(message)
            self.next_btn['state'] = 'disabled'
            self.exit_btn['state'] = 'disabled'
            prn = False

        #prn = True
        self.tk.update_idletasks()
        if prn: print(f' >>> {state}')
        time.sleep(GUI_DELAY)

    def __put_decision(self, dec:int):
        """ returns human decision (decision button pressed) """
        message = QMessage(type='decision', data=dec)
        self.queO.put(message)
        self.__set_dec_cash_val()
        self.__set_dec_btn_act()

    ### players frames methods ****************************************************************** players frames methods

    def __upd_pl_won(self, plix:int, won):
        """ updates player tot won """
        self.pl_won[plix] += int(won)
        self.plx_elD[plix]['lblL'][0]['text'] = self.pl_won[plix]

    def __upd_plcsh(self, pl_ix:int, cash:int=None, cash_cs:int=None):
        """ updates player cash
        cash and cash_cs - for True do not update """
        if cash is None: cash = '-'
        if cash_cs is None: cash_cs = '-'
        if cash is not True:     self.plx_elD[pl_ix]['lblL'][4]['text'] = cash
        if cash_cs is not True:  self.plx_elD[pl_ix]['lblL'][5]['text'] = cash_cs

    def __set_pl_active(self, plix:int, a=True):
        self.plx_elD[plix]['lblL'][4]['fg'] = 'black' if a else 'gray36'
        self.plx_elD[plix]['lblL'][5]['fg'] = 'red4' if a else 'gray36'

    def __set_button(self, i:int=None):
        set_image(self.plx_elD[i]['lblL'][2], self.dealer_img)
        other = list(range(len(self.players)))
        other.pop(i)
        for ix in other:
            set_image(self.plx_elD[ix]['lblL'][2], self.nodealer_img)

    ### table frame methods ************************************************************************ table frame methods

    # updates my cards
    def __upd_myc(self, ca :str=None, cb :str=None):
        set_image(self.myc_lblL[0], self.cards_imagesD[ca])
        set_image(self.myc_lblL[1], self.cards_imagesD[cb])

    def __upd_tblc(self, cl:Optional[List]=None):
        """ updates table cards (self.tcards) list """

        # update list
        if not cl:
            self.tcards = []
        else:
            self.tcards += cl

        # update GUI
        cl = [] + self.tcards # copy
        cl += [None]*(5-len(cl))
        for ix in range(5):
            set_image(self.tblc_lblL[ix], self.cards_imagesD[cl[ix]])

    def __upd_tcash(
            self,
            street: Optional[int]=    None,
            pot: Optional[int]=      None):
        """ updates table cash """
        if street is None: street = '-'
        if pot is None: pot = '-'
        self.tcsh_lblD['street']['text'] = f'street:{street}'
        self.tcsh_lblD['pot']['text'] = f'POT:{pot}'

    ### decision frame methods ****************************************************************** decision frame methods

    def __set_dec_cash_val(self, val:Optional[List]=None):
        """ sets $ values of cash labels """
        if not val: val = ['-']*len(self.gc.table_moves)
        for lbl,v in zip(self.dec_cashL,val):
            lbl['text'] = v

    def __set_dec_btn_act(self, act:Optional[List[bool]]=None):
        """ sets state of buttons """
        if not act:
            act = [False]*len(self.gc.table_moves)
        for ix in range(len(self.dec_btnL)):
            self.dec_btnL[ix]['state'] = 'normal' if act[ix] else 'disabled'

    def __set_dec_math_text(self, phase:int):
        """ sets text math labels """
        for lbl,nm in zip(self.dec_mathL, self.move_math[phase]):
            lbl['text'] = nm