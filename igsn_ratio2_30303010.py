#xinyuanyan
#Dec,2020
#Key words: ToM, penny-competitive game, complex network, computational modeling


#assuming you are in the github folder change the path - not relevant if tomsup is installed via. pip
import math 
import os
print(os.getcwd())
os.chdir("..") # go out of the tutorials folder
print(os.getcwd())

import tomsup as ts


#creat ToM agents
ts.valid_agents()
#ts.TOM()
#Create a 2-TOM agent with volatility -2, a low temperature -4, a bias of 0.5 and a dilution of 0.4.


#We can extract the given parameter values of the agents
#print(sir_TOM0.get_parameters())
#Get the competitive penny game payoff matrix
penny = ts.PayoffMatrix("penny_competitive")

def exponential_func(x, aval): #define exponential function here
    y=math.pow(aval, x)
    return y





import numpy as np
import matplotlib.pyplot as plt
# generate an array with specific number of ToMs
tom_zero_three = [30,30,30,10]
tomarray = []
tom0 = np.zeros(tom_zero_three[0])
tom1 = np.ones(tom_zero_three[1])
tom2 = np.ones(tom_zero_three[2]) * 2
tom3 = np.ones(tom_zero_three[3]) * 3

tomarray = np.concatenate((tom0, tom1, tom2, tom3))

np.random.shuffle(tomarray)  #randomized
tommat = tomarray.reshape(10, 10)

print(tommat)  #check now

#visualize the tomat

plt.imshow(tommat)

plt.colorbar()

len(tommat)
print(len(tommat))
a = range(len(tommat))
print(a)
range(0,10)
print(tommat[0][9])
print(tommat[1][7])
#a[9]
#print(tommat[0,a[9]])

#interup, so load the current tommat first

tommat = np.load('tommat.npy')

#now let the agent in each cell activate and interact!

totalsteps = 1000  # 10000 steps with newborns
  
iB = 5  # intial benefits for every agent in 0 step 0 round
        
total_round = 10 # in 1 step, how many rounds will the two agents interact?
        
tomagents = ['0-TOM', '1-TOM', '2-TOM', '3-TOM']

#setting the potential energy-cost ratio

energy_a = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1]#y = a^x (a-value)

all_type_results = np.zeros(shape=(len(energy_a),int(totalsteps),len(tomagents)))
fB = iB*np.ones(shape=(10, 10))  # final benefits, we need to update it ! but with initial 1 units in each grid cell

for EC in [0]:#range(len(energy_a)):
    
    saveEC = energy_a[EC]
    
    
    tomagent_sv_ratio = np.zeros(shape=(int(totalsteps),len(tomagents)))
    
    for step in range(47,101,1):#range(totalsteps):
    
        # now, xinyuan, you need to define the specific grid, then find their neighbors to interact and collect data
        # now interact and collect data
    
        checkstep = step
    

        
        ##########
        #agent_colum = 0
        ##########
    
        for agent_colum in [0]:
            
            print('now the col', str(agent_colum))
    
            for agent_row in range(len(tommat)):
                
                print('now the row', str(agent_row))
    
                # -----------------------1st if if if----------------------#
    
                if  agent_colum==0 and agent_row == 0:# can only interact with three agents
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row][agent_colum + 1], tommat[agent_row + 1][agent_colum],
                                             tommat[agent_row + 1][agent_colum + 1]]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    # !!!!!!!!!!
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum] + np.mean(final_out_central_payoff)
                    fB[agent_row][agent_colum + 1] = fB[agent_row][agent_colum + 1]+np.mean(final_out_payoff[0])
                    fB[agent_row + 1][agent_colum] =  fB[agent_row + 1][agent_colum]+np.mean(final_out_payoff[1])
                    fB[agent_row + 1][agent_colum + 1] = fB[agent_row + 1][agent_colum + 1]+np.mean(final_out_payoff[2])
                
                print('end row', str(agent_row))
    
                
                # -----------------------2nd if if if----------------------#
    
                # can only interact with three agents
                if agent_colum==0 and agent_row == len(tommat) - 1:
                    
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    
                    
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row - 1][agent_colum], tommat[agent_row - 1][agent_colum + 1],
                                             tommat[agent_row][agent_colum + 1]]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    # !!!!!!!!!!
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum] + np.mean(final_out_central_payoff)
                    fB[agent_row - 1][agent_colum] =fB[agent_row - 1][agent_colum]+np.mean(final_out_payoff[0])
                    fB[agent_row - 1][agent_colum + 1] = fB[agent_row - 1][agent_colum + 1]+np.mean(final_out_payoff[1])
                    fB[agent_row][agent_colum + 1] = fB[agent_row][agent_colum + 1]+np.mean(final_out_payoff[2])
                
                print('end row', str(agent_row))
                
    
                # 现在开始考虑外面的，但是夹在中间的格子了。
    
               
    
                # -----------------------3rd if if if----------------------#
    
                # can only interact with five agents
                if agent_colum==0 and agent_row in range(1,len(tommat)-1):
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row - 1][agent_colum], tommat[agent_row + 1][agent_colum],
                                             tommat[agent_row - 1][agent_colum +
                                                                   1], tommat[agent_row + 1][agent_colum + 1],
                                             tommat[agent_row][agent_colum + 1]
                                             ]
                    # check
                    print(interact_tomagent_idx)
    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
    
    
                    # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum]+np.mean(final_out_central_payoff)
                    fB[agent_row - 1][agent_colum] = fB[agent_row - 1][agent_colum]+np.mean(final_out_payoff[0])
                    fB[agent_row + 1][agent_colum] = fB[agent_row + 1][agent_colum]+np.mean(final_out_payoff[1])
                    fB[agent_row - 1][agent_colum + 1] = fB[agent_row - 1][agent_colum + 1] + np.mean(final_out_payoff[2])
                    fB[agent_row + 1][agent_colum + 1] = fB[agent_row + 1][agent_colum + 1]+ np.mean(final_out_payoff[3])
                    fB[agent_row][agent_colum + 1] = fB[agent_row][agent_colum + 1]+np.mean(final_out_payoff[4])
                print('end row', str(agent_row))
    
    ######################################################################################################################
    
    
    # now, xinyuan, let us continue
        for agent_colum in [len(tommat)-1]:
            print('now the col', str(agent_colum))
    
            for agent_row in range(len(tommat)):
                # -----------------------1st if if if----------------------#
    
                if  agent_colum==len(tommat)-1 and agent_row == 0:# can only interact with three agents
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row][agent_colum-1], tommat[agent_row+1][agent_colum],
                                          tommat[agent_row+1][agent_colum-1]]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum]+np.mean(final_out_central_payoff)
                    fB[agent_row][agent_colum-1] = fB[agent_row][agent_colum-1]+np.mean(final_out_payoff[0])
                    fB[agent_row+1][agent_colum] = fB[agent_row+1][agent_colum] +np.mean(final_out_payoff[1])
                    fB[agent_row+1][agent_colum-1] = fB[agent_row+1][agent_colum-1]+np.mean(final_out_payoff[2])
                
                print('end row col9', str(agent_row))
    
                # -----------------------2nd if if if----------------------#
    
                # can only interact with three agents
                if agent_colum==len(tommat)-1 and agent_row == len(tommat) - 1:
                    
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    print('play the last 99999999!!!')
                    
                    
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row-1][agent_colum], tommat[agent_row-1][agent_colum-1],
                                              tommat[agent_row][agent_colum-1]]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    # !!!!!!!!!!
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                    fB[agent_colum][agent_row] =fB[agent_colum][agent_row]+ np.mean(final_out_central_payoff)
                    fB[agent_row - 1][agent_colum] = fB[agent_row - 1][agent_colum]+np.mean(final_out_payoff[0])
                    fB[agent_row - 1][agent_colum - 1] = fB[agent_row - 1][agent_colum - 1] +np.mean(final_out_payoff[1])
                    fB[agent_row][agent_colum - 1] = fB[agent_row][agent_colum - 1]+np.mean(final_out_payoff[2])
                
                print('end row col9', str(agent_row))
                
    
                # 现在开始考虑外面的，但是夹在中间的格子了。
                # -----------------------3rd if if if----------------------#
    
                # can only interact with five agents
                if agent_colum==len(tommat)-1 and agent_row in range(1,len(tommat)-1):
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    print('play the middle1!!!!!!!!!!!')
                    
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
    
                    interact_tomagent_idx = [tommat[agent_row-1][agent_colum], tommat[agent_row+1][agent_colum],
                                              tommat[agent_row-1][agent_colum -
                                                                  1], tommat[agent_row+1][agent_colum-1],
                                              tommat[agent_row][agent_colum-1]
                                              ]
                    # check
                    print(interact_tomagent_idx)
    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
    
    
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum] +np.mean(final_out_central_payoff)
                    fB[agent_row-1][agent_colum] = fB[agent_row-1][agent_colum]+ np.mean(final_out_payoff[0])
                    fB[agent_row+1][agent_colum] = fB[agent_row+1][agent_colum]+np.mean(final_out_payoff[1])
                    fB[agent_row-1][agent_colum-1] = fB[agent_row-1][agent_colum-1]+np.mean(final_out_payoff[2])
                    fB[agent_row+1][agent_colum-1] = fB[agent_row+1][agent_colum-1]+np.mean(final_out_payoff[3])
                    fB[agent_row][agent_colum-1] =fB[agent_row][agent_colum-1] + np.mean(final_out_payoff[4])
                    
                print('end row col9', str(agent_row))
    
    
    ###################################################################################################
    
    
    # now, xinyuan, let us continue
        for agent_colum in range(1,len(tommat)-1):
           
            for agent_row in [0]:
                # -----------------------1st if if if----------------------#
    
                if  agent_row == 0:# can only interact with three agents
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row][agent_colum-1], tommat[agent_row][agent_colum+1],
                                              tommat[agent_row+1][agent_colum -
                                                                  1], tommat[agent_row+1][agent_colum+1],
                                              tommat[agent_row+1][agent_colum]
                                              ]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum]+np.mean(final_out_central_payoff)
                    fB[agent_row][agent_colum-1] = fB[agent_row][agent_colum-1]+np.mean(final_out_payoff[0])
                    fB[agent_row][agent_colum+1] = fB[agent_row][agent_colum+1]+np.mean(final_out_payoff[1])
                    fB[agent_row+1][agent_colum-1] =fB[agent_row+1][agent_colum-1]+ np.mean(final_out_payoff[2])
                    fB[agent_row+1][agent_colum+1] = fB[agent_row+1][agent_colum+1]+np.mean(final_out_payoff[3])
                    fB[agent_row+1][agent_colum] = fB[agent_row+1][agent_colum]+np.mean(final_out_payoff[4])
                
                print('end row col9', str(agent_row))
    
    
    # now, xinyuan, let us continue
        for agent_colum in range(1,len(tommat)-1):
           
            for agent_row in [len(tommat)-1]:
                # -----------------------1st if if if----------------------#
    
                if  agent_row == len(tommat)-1:# can only interact with three agents
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row][agent_colum-1], tommat[agent_row][agent_colum+1],
                                              tommat[agent_row-1][agent_colum -
                                                                  1], tommat[agent_row-1][agent_colum+1],
                                              tommat[agent_row-1][agent_colum]
                                              ]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    #
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum] +np.mean(final_out_central_payoff)
                    fB[agent_row][agent_colum-1] = fB[agent_row][agent_colum-1]+np.mean(final_out_payoff[0])
                    fB[agent_row][agent_colum+1] = fB[agent_row][agent_colum+1]+np.mean(final_out_payoff[1])
                    fB[agent_row-1][agent_colum-1] =fB[agent_row-1][agent_colum-1]+ np.mean(final_out_payoff[2])
                    fB[agent_row-1][agent_colum+1] = fB[agent_row-1][agent_colum+1]+np.mean(final_out_payoff[3])
                    fB[agent_row-1][agent_colum] = fB[agent_row-1][agent_colum]+np.mean(final_out_payoff[4])
                
                print('end row col9', str(agent_row))
                
                
    
    #####################最中间的所有格子#####################
    
    
    
    # now, xinyuan, let us continue
        for agent_colum in range(1,len(tommat)-1):
           
            for agent_row in range(1,len(tommat)-1):
                # -----------------------1st if if if----------------------#
    
               
                    core_tomagent_idx = tommat[agent_row][agent_colum]
    
                    interact_tomagent_idx = [tommat[agent_row-1][agent_colum], tommat[agent_row+1][agent_colum],
                                             tommat[agent_row-1][agent_colum -1], tommat[agent_row+1][agent_colum-1],
                                             tommat[agent_row][agent_colum-1], tommat[agent_row][agent_colum+1],
                                             tommat[agent_row-1][agent_colum +1], tommat[agent_row+1][agent_colum+1]
                                             ]
                    # check
                    print(interact_tomagent_idx)
    
                    ###############################################################
                    final_out_payoff = []
                    final_out_central_payoff = []
                    
                    for i in range(len(interact_tomagent_idx)):
    
                        all_agents = [tomagents[int(core_tomagent_idx)],
                                      tomagents[int(interact_tomagent_idx[i])]]
                        print(all_agents)
                        all_params = [{}, {}]
    
                        # We add the save_history to all parameter sets, in order to make sure we can get the internal states of all agents
                        for d in all_params:
                            d['save_history'] = True
    
                        # Now we create the group and set the tournament environment
                        group = ts.AgentGroup(all_agents, all_params)
    
                        group.set_env(env='round_robin')
    
                        # Finally, we make the group compete 20 simulations of 30 rounds
                        group.compete(p_matrix=penny, n_rounds=total_round,
                                      n_sim=total_round, save_history=True)
                        df = group.head(total_round)
    
                        # get the mean payoff of each agent
                        final_out_perstep = df.describe()
                        final_out_central_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent0)  # core agent
                        # its arounding agents
                        final_out_payoff.append(
                            final_out_perstep.loc['mean'].payoff_agent1)
    
                    # now let us update the fB for each agent mentioned above
                # now let us update the fB for each agent mentioned above
                # now let us update the fB for each agent mentioned above
                    fB[agent_row][agent_colum] = fB[agent_row][agent_colum]+np.mean(final_out_central_payoff)
                    fB[agent_row-1][agent_colum] = fB[agent_row-1][agent_colum]+np.mean(final_out_payoff[0])
                    fB[agent_row+1][agent_colum] =fB[agent_row+1][agent_colum]+ np.mean(final_out_payoff[1])
                    
                    fB[agent_row-1][agent_colum-1] = fB[agent_row-1][agent_colum-1] +np.mean(final_out_payoff[2])
                    fB[agent_row+1][agent_colum-1] = fB[agent_row+1][agent_colum-1] +np.mean(final_out_payoff[3])
                    fB[agent_row][agent_colum-1] = fB[agent_row][agent_colum-1] +np.mean(final_out_payoff[4])
                    
                    fB[agent_row][agent_colum+1] =fB[agent_row][agent_colum+1]+ np.mean(final_out_payoff[5])
                    fB[agent_row-1][agent_colum+1] = fB[agent_row-1][agent_colum+1]+np.mean(final_out_payoff[6])
                    fB[agent_row+1][agent_colum+1] =fB[agent_row+1][agent_colum+1]+np.mean(final_out_payoff[7])
                
               
    ######################现在在fB的基础上，要减去各个agent消耗的能量######################
        current_energy = fB
        #get the ratio for all grid cells
        tomlevel = tommat
        energy_cost_ratio = np.zeros(shape=(10, 10))
        for tom_row in range(len(tommat)):
            for tom_col in range(len(tommat)):
                energy_cost_ratio[tom_row][tom_col]=exponential_func(tommat[tom_row][tom_col], energy_a[EC])
        
        #save the updated energy
        this_step_enery = current_energy-current_energy*energy_cost_ratio/100
        fB = this_step_enery #updated fB val
        
        #to see who die? if die, he will be replaced by the agent who has maxium value
        r, c = np.where(this_step_enery == np.max(this_step_enery))
        print(r,c)
        max_agent = tommat[r[0]][c[0]]
        tommat_new = np.zeros(shape = (10,10))
        
        for tom_row in range(len(tommat)):
            for tom_col in range(len(tommat)):
                if this_step_enery[tom_row][tom_col]<0:
                    tommat[tom_row][tom_col] = max_agent
                    
        #tommat = tommat_new#update tommat for next step
        #collect the survived ratio
        tomagent_sv_ratio[step][0]= np.sum(tommat == 0)/100
        tomagent_sv_ratio[step][1]= np.sum(tommat == 1)/100
        tomagent_sv_ratio[step][2]= np.sum(tommat == 2)/100
        tomagent_sv_ratio[step][3]= np.sum(tommat == 3)/100
        
    
        #plot the results for each a (energy_cost ratio a-val)
        all_type_results[EC][step][0]= np.sum(tommat == 0)/100
        all_type_results[EC][step][1]= np.sum(tommat == 1)/100
        all_type_results[EC][step][2]= np.sum(tommat == 2)/100
        all_type_results[EC][step][3]= np.sum(tommat == 3)/100
        
        #save data in current working path
        np.save('ratio2_all_type_results.npy',all_type_results)
        np.save('ratio2_tomagent_sv_ratio.npy',tomagent_sv_ratio)
        np.save('ratio2_tommat.npy',tommat)
        np.save('ratio2_fB.npy',fB)
        np.save('ratio2_whichEC',saveEC)
        np.save('ratio2_whichstep',checkstep)
        
        #f=open("data/which_step.txt",'a')# #若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原

                                            #内容之后写入。可修改该模式（'w+','w','wb'等）
        #f.write()
        
        
                        

#####################save key data##########################3
                