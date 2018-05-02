# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

SCREEN_WIDTH  = 600
SCREEN_HEIGHT = 400
BINSIZE = 40 # Pixels per bin, we can try tune this size
ACTION_STATE = 2 # 0 or 1
GRAVITY_STATE = 2 # 1 or 4
VSTATES = 6 # Velocity states [-3,-2,-1,0,1,2], can be tuned

GAMMA = 0.7 # Discount factor, can be tuned as well
EPS_FACTOR = 0.002 # Start e-greedy factor, can be tuned

EPOCHS = 1000

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.first_state = True
        self.gravity = None

        self.last_D = None
        self.last_B = None
        self.last_V = None

        # state: (horizontal dist to tree, vertical dist to tree, dist to tree bottom, vel, gravity)
        self.Q = np.zeros((ACTION_STATE,SCREEN_WIDTH/BINSIZE,2*SCREEN_HEIGHT/BINSIZE,VSTATES,GRAVITY_STATE)) # Q value matrix
        self.k = np.zeros((ACTION_STATE,SCREEN_WIDTH/BINSIZE,2*SCREEN_HEIGHT/BINSIZE,VSTATES,GRAVITY_STATE)) # number of times action a has been taken from state s

        self.epoch = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.first_state = True
        self.gravity = None

        self.last_D = None
        self.last_B = None
        self.last_V = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        self.epoch += 1

        if self.last_reward is None:
            self.last_state = state
            return 0
        else:
            if self.first_state is True:
                self.gravity = 1 if state['monkey']['vel'] != -1 else 0
                self.first_state = False

        # current state
        D = state['tree']['dist'] / BINSIZE
        if D < 0:
            D = 0
        B = (state['monkey']['bot'] - state['tree']['bot']) / BINSIZE
        V = state['monkey']['vel'] / 10
        G = self.gravity

        if V > 2:
            V = 2
        elif V < -3:
            V = -3

        def default_action(p=0.5):
            return 1 if npr.rand() < p else 0

        new_action = 0
        if not self.last_action == None:
            max_Q = np.max(self.Q[:,D,B,V,G])
            new_action = 1 if self.Q[1][D,B,V,G] > self.Q[0][D,B,V,G] else 0

            # epsilon-greedy
            if self.k[new_action][D,B,V,G] > 0:
                eps = EPS_FACTOR / self.k[new_action][D,B,V,G] / (1 + 2.7 ** ((EPOCHS * 0.8 - self.epoch)/EPOCHS))
            else:
                eps = EPS_FACTOR
            if (npr.rand() < eps):
                new_action = default_action()

            ALPHA = 1/self.k[self.last_action][self.last_D,self.last_B,self.last_V,G]

            # update Q
            self.Q[self.last_action][self.last_D,self.last_B,self.last_V,G] += ALPHA*(self.last_reward+GAMMA*max_Q-self.Q[self.last_action][self.last_D,self.last_B,self.last_V,G])

        self.last_action = new_action
        self.last_state  = state
        self.k[new_action][D,B,V,G] += 1

        self.last_D = D
        self.last_B = B
        self.last_V = V


        return new_action


    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        print swing.score,
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, EPOCHS, 10)
    print hist
    print max(hist)

    # Save history.
    np.save('hist',np.array(hist))
