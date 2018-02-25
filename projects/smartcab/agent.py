import random
import math
import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        self.epsilon = self.epsilon  - 0.05
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        #state = None
        #defining state as tuple for Quest 5
        state =( waypoint, inputs['light'], inputs['oncoming'])

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        print ('max')
        #print(max(self.Q.iteritems(), key = operator.itemgetter(1))[0])
        #reference https://stackoverflow.com/questions/42044090/return-the-maximum-value-from-a-dictionary
        # https://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key/26871933
        #maxQ = max(self.Q.values())
        print ('dict action for given state' ,self.Q[state].values())
        #for x in [self.Q[state] for x in self.Q]
        #maxQ = max(self.Q, key = lambda x:x[state])
        #q = [self.getQ]
        # maxQ value 
        maxQval = max(self.Q[state][d] for d in self.Q[state].keys())
        #maxQ  = max(seq)
        #maxQ = 1*self.Q[state].values()
        
        #maxQ = max(self.Q[state])
        #maxkey = max(self.Q, key = self.Q.get)
       # print('maxkey', maxkey)
        #maxQ = self.Q.get(max(self.Q.values())
        #maxQ= self.Q.get('')
        print('max',maxQval)
        #reference https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
        #https://www.quora.com/Is-there-a-way-to-extract-specific-keys-and-values-from-an-ordered-dictionary-in-Python
        for key, value in self.Q[state].items():
            if value == maxQval:
                print('in if ' ,self.Q[state][key] )
               # maxQ = self.Q[state]
                maxQ = self.Q[state][key] 
                print('maxq', maxQ)
 
        #print('max of',maxQ[state][act])
      #  maxQ = max(self.Q.[state][act])
       # maxQ = max(self.Q, key = self.Q.get)
      #  maxQ = max(self.Q, key = lambda x:x[self.Q[state][act]])
       # print(maxQ)

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        #initializing Q table for each possible state
        
        #self.Q = {}
        #for way in ['left', 'right', 'forward']:
        #    for lgt in ['green' ,'red']:
        #        for onc in [None, 'forward', 'left', 'right']:
        #            for act in self.env.valid_actions:
        #                #self.Q[(way, lgt,onc, act)] = 0.0
        #                 self.Q[(way, lgt,onc)] = {act, 0.0} 
        print('key',self.Q.keys())
        #self.Q.keys()
        if self.learning == True:
            print(state)
            
            if state in self.Q.keys():
                print('FOUND!!!!!!!')    
                
            if state not in self.Q.keys():
                #act= self.choose_action(state)
                self.Q[state] = dict()
                #print('acttest')
                #print(act)
                print(self.Q)
          
               # self.Q = {'key':'value'}
                print(self.Q)
                print(state)
                for act in self.env.valid_actions:
                    self.Q[state][act]=  0.0 
                print(self.Q.viewitems)
                
            print('what is in self.Q')   
            print(self.Q)       
              
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        print("state")
        print(self.state)
       # action = None
#simulation for basic driving agent
        #action = self.valid_actions
#reference - https://stackoverflow.com/questions/306400/how-to-randomly-select-an-item-from-a-list


        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        if self.learning == True:
          #  action = random.choice(self.valid_actions, self.epsilon)
          if self.epsilon > random.random():
              action = random.choice(self.valid_actions)
          else:
              print('in chosose action max',self.get_maxQ(state) )
              #hQ = self.get_maxQ(state)
              #action = random.choice(self.get_maxQ(state))
              action = random.choice(self.get_maxQ(state))
              #action = random.choice(hQ)
        else:
            action = random.choice(self.valid_actions)
            print( "actions valid " )
            print(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        if self.learning == True:
            print('maxQ is')
            print(self.get_maxQ(state))
            
        # value iteration formula is Q(s,a) = (1-alpha)Q(s,a) + alpha(reward + gamma(maxQ(s',a'))
            print('what is q value for current state')
          #  print(self.Q[state][action] )
            self.Q[state][action]= (1 -self.alpha)*(self.Q[state][action])  + self.alpha*(reward)
            #self.Q.values()[state] = (1-self.alpha)*self.Q[state][action] 
           #+ self.get_maxQ(state)
           # self.Q.values()[state] = (1-self.alpha)*self.Q[state][action] 
            #self.Q.values()[state] = reward + self.get_maxQ(state)
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning = True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,display=False, update_delay = .01, log_metrics = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10)


if __name__ == '__main__':
    run()
