## pologic - poker logic of Table & (table) Player

Pologic stores the logic of poker game, where objects such as
**Table**, **Player**, **Hand History**, and **Card Deck** are defined.
---

### Table & Player

The Table has players and runs a single poker game.
The Table has a name and is initialized with a list of Player IDs.

![](../images/table_players.png)

##### Running a Hand:

The initial order of Players (given during the Table initialization) is used for the first hand played.
For every subsequent hand, players are rotated.

The Table builds a **Hand History** (HH) with events while playing the hand.
Each Player is occasionally asked to make a move (a decision) for which they may (or should) use HH data.

##### QPTable & QPPlayer:

QPTable is a table that uses QPPlayer instead of PPlayer. QPTable is a Process,
and QPPlayer uses queues to communicate with the decision-making object (DMK - Decision MaKer) in **pypoks**.
---

### podeck - poker Cards Deck

Cards may be represented in 3 types:
- int 0-52, where 52 represents a 'pad' (an unknown card)
- tuple (int,int) (0-13,0-3) (pad is (13,0))
- str ‘9C’, figures: 2..9TJDKAX; colors: SHDC (pad is XS)

Each representation has its advantages:
- integers are simple
- tuples are used for efficient card rank computation
- strings are easy to read and somewhat standard for poker notation
---

### Hand History

Hand History (HH) is a list of events that occured during a single poker hand run.
HH is complete, and the hand may be replayed using the given HH.

A State is represented as a ```Tuple[str,Tuple]```, consisting of a name and tuple with data.
Below is an example of an HH:

    HST ('table', 0)
    T$$ (0, 0, 0)
    TST (0,)
    POS ('pl0', 0)
    POS ('pl1', 1)
    PSB ('pl0', 2)
    PBB ('pl1', 5)
    T$$ (7, 7, 5)
    PLH ('pl1', '9S', 'KS')
    PLH ('pl0', 'TC', 'TH')
    TST (1,)
    T$$ (7, 7, 5)
    MOV ('pl0', 1, 0, (498, 2, 2))
    T$$ (7, 7, 5)
    PRS ('pl0', -2, 'muck')
    PRS ('pl1', 2.0, 'not_shown')
    HFN ('table', 0)

Detailed description of the state format may be found in pologic/hand_history.py