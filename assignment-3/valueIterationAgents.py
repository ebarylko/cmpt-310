# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np

from learningAgents import ValueEstimationAgent
import collections
import functools as ft

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        def calc_new_state_val(state):
            next_actions = self.mdp.getPossibleActions(state)
            calc_q_value = ft.partial(self.computeQValueFromValues,
                                      state)
            #print("The next actions are ", next_actions)
            new_state_val = 0 if self.mdp.isTerminal(state) else max(map(calc_q_value, next_actions))
            #print("The new value for state {} is {}".format(state, new_state_val))
            return state, new_state_val

        def update_state(state, new_val):
            self.values[state] = new_val

        def update_values_of_each_state(all_states):
            states_and_new_values = list(map(calc_new_state_val, all_states))
            #print("The states and their new values: ", list(states_and_new_values))
            for (state, new_val) in states_and_new_values:
                update_state(state, new_val)

        for _ in range(self.iterations):
            update_values_of_each_state(self.mdp.getStates())


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        def calc_state_transition_reward(initial_state, initial_action,
                                         new_state_and_transition_prob):
            new_state, transition_prob = new_state_and_transition_prob
            #if initial_state == (0, 1):
                #print("The new state is {}, the probability of reaching it is {}, and the reward is {}".format(new_state, transition_prob, self.mdp.getReward(initial_state, initial_action, new_state)))
                #print("The value of the new state is ", self.getValue(new_state))
            return transition_prob * (self.mdp.getReward(initial_state, initial_action, new_state) + self.discount * self.getValue(new_state))

        calc_reward = ft.partial(calc_state_transition_reward, state, action)
        #print("The starting state and action are {} and {}".format(state, action))
        #print("The values of all of the possible actions are ", list(map(calc_reward, self.mdp.getTransitionStatesAndProbs(state, action))))
        return sum(map(calc_reward, self.mdp.getTransitionStatesAndProbs(state, action)))



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #print("The state is ", state)
        next_actions = self.mdp.getPossibleActions(state)
        if len(next_actions) == 0:
            return None

        q_vals_of_actions = list(map(ft.partial(self.computeQValueFromValues, state),
                                     next_actions))
        #print("The q-values are ", q_vals_of_actions)
        idx_of_best_action = np.argmax(q_vals_of_actions)
        #print("The idx is ", idx_of_best_action)
        #print("The best action is ", next_actions[idx_of_best_action])
        return next_actions[idx_of_best_action]



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


