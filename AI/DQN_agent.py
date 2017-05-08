'''
Deep Q-learning approach for robotic arm and camera coordination
'''

import random
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import tensorflow as tf

import UDP

import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists
    that get returned as another list of dictionaries with each key corresponding to either
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size) :
        indices = random.sample(np.arange(len(self.states)), min(size,len(self.states)) )
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal) :
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)

        self.currentPosition += 1

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def loadModel(self):
        self.model = load_model('savedModels/model_DQN.h5')
        # self.targetModel = load_model('savedModels/model_DQN.h5')

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 32
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)

                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)


#main

with tf.device('/gpu:0'):

    start_time = time.time()

    epochs = 3000
    steps = 2000 #600
    updateTargetNetwork = 200 #500
    explorationRate = 1
    minibatch_size = 32
    learnStart = 32
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000

    explorationFactor = 0.995

    Train = False

    if not Train:
        epochs = 1000
        steps = 500
        explorationRate = 0.05

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False

    unity = UDP.UDP()

    deepQ = DeepQ(6, 12, memorySize, discountFactor, learningRate, learnStart)
    #deepQ = DeepQ(3, 27, memorySize, discountFactor, learningRate, learnStart)

    #deepQ.initNetworks([30,30,30])
    #deepQ.initNetworks([30,30])
    if Train:
        deepQ.initNetworks([300,300])
    else:
        deepQ.loadModel()

    stepCounter = 0
    realEpoch = epochs - 1

    plt.ion()
    fig1 = plt.figure(1)
    plt.xlim(0, realEpoch)
    plt.xlabel('Epochs')
    plt.ylabel('Successes')
    plt.plot([], [], color='blue')

    plt.ion()
    fig2 = plt.figure(2)
    plt.xlim(0, realEpoch)
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward')
    plt.plot([], [], color='blue')

    plt.ion()
    fig3 = plt.figure(3)
    plt.xlim(0, realEpoch)
    plt.xlabel('Epochs')
    plt.ylabel('Average Q-Value')
    plt.plot([], [], color='blue')

    plt.ion()
    fig4 = plt.figure(4)
    plt.xlim(0, realEpoch)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.plot([], [], color='blue')

    plt.show(block=False)

    allActions = []
    for item in itertools.product([-1,1], repeat = 3):
        allActions.append(item)

    allActions.append((-1, -1, 0))
    allActions.append((-1, 1, 0))
    allActions.append((1, -1, 0))
    allActions.append((1, 1, 0))
    # allActions.remove((0,0,0))

    print allActions

    numEpoch = list()
    numTimesteps = list()
    averageReward = list()
    averageQvalue = list()
    epsilon = list()

    numDones = 0
    lastDistance = 0

    # number of reruns
    for epoch in xrange(epochs):

        unity.newEpisode()
        observation = unity.newObservation()

        initial_observation = np.array([observation[0], observation[1], observation[2], observation[0], observation[1], observation[2]])

        totalReward = 0
        totalQvalue = 0

        # number of timesteps
        for t in xrange(steps):
            if t != 0 :
                unity.noEpisode()

            if t == 0:
                qValues = deepQ.getQValues(initial_observation)
                observation = initial_observation
            else:
                qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)
            newObservation_aux, done = unity.sendAction(allActions[action])
            temp_observation = observation;

            k=0
            while np.array_equal(np.array([observation[0], observation[1], observation[2]]), np.array([newObservation_aux[0], newObservation_aux[1], newObservation_aux[2]])):
                unity.noEpisode()
                newObservation_aux, done = unity.sendAction(allActions[action])
                # print "bucle ", k
                # print "Action: ", allActions[action]
                # print "Observation: ", observation
                # print "newObservation: ", newObservation
                if k == 20:
                    break
                k += 1

            distance = np.sqrt(newObservation_aux[0]**2 + newObservation_aux[1]**2 + newObservation_aux[2]**2)
            if (distance < lastDistance):
                reward = 0
                #reward = 0.5
                #reward = np.exp(-1 * distance)
                #reward = 1 - distance
            else:
                 reward = -1

            newObservation = np.array([newObservation_aux[0], newObservation_aux[1], newObservation_aux[2],
                                       newObservation_aux[0] - observation[0], newObservation_aux[1] - observation[1], newObservation_aux[2] - observation[2]])

            if done:
                print "Sucess!"
                numDones += 1

            if (t == steps-1) and not done:
                print "Failed. Time out"
                done = True

            totalReward += reward
            totalQvalue += deepQ.getMaxQ(qValues)

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if Train:
                if stepCounter >= learnStart:
                    if stepCounter <= updateTargetNetwork:
                        deepQ.learnOnMiniBatch(minibatch_size, False)
                    else :
                        deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation
            lastDistance = distance

            if done:
                '''
                last100Scores[last100ScoresIndex] = t
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print "Episode ",epoch," finished after {} timesteps".format(t+1)
                else :
                    print "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores))
                '''
                plt.figure(1)
                numEpoch.append(epoch)
                numTimesteps.append(numDones)
                plt.plot(numEpoch, numTimesteps, 'b-', linewidth=0.1)
                fig1.canvas.draw()
                if Train:
                    plt.savefig("train/steps_episodes.png")
                else:
                    plt.savefig("test/steps_episodes.png")

                if (totalReward == 0):
                    averageReward.append(0)
                else :
                    averageReward.append(totalReward)
                    #averageReward.append(totalReward/(t+1))

                plt.figure(2)
                plt.plot(numEpoch, averageReward, 'b-', linewidth=0.1)
                fig2.canvas.draw()
                if Train:
                    plt.savefig("train/reward_episodes.png")
                else :
                    plt.savefig("test/reward_episodes.png")

                plt.figure(3)
                averageQvalue.append(totalQvalue/(t+1))
                plt.plot(numEpoch, averageQvalue, 'b-', linewidth=0.1)
                # z = np.polyfit(numEpoch, averageQvalue, 1)
                # p = np.poly1d(z)
                # plt.plot(numEpoch,p(numEpoch),"r--", linewidth=1.5)
                fig3.canvas.draw()
                if Train:
                    plt.savefig("train/Qvalue_episodes.png")
                else :
                    plt.savefig("test/Qvalue_episodes.png")

                plt.figure(4)
                epsilon.append(explorationRate)
                plt.plot(numEpoch, epsilon, 'b-', linewidth=0.1)
                fig4.canvas.draw()
                if Train:
                    plt.savefig("train/epsilon.png")
                else :
                    plt.savefig("test/epsilon.png")

                plt.show(block=False)

                print "Episode: {} Timestep: {} Epsilon: {} Total Reward: {} Average Q-Value: {} Observation: {}".format(epoch, t+1, explorationRate, totalReward, totalQvalue/(t+1), newObservation)
                print "Success rate: {:.1%}".format(float(numDones)/float(epoch+1))
                print "---------------------------------------------------------------------------"

                if Train:
                    deepQ.targetModel.save('savedModels/model_DQN.h5')

                break
                #sys.exit(0)

            stepCounter += 1
            if Train:
                if stepCounter % updateTargetNetwork == 0:
                    deepQ.updateTargetNetwork()
                    #print "updating target network"

        explorationRate *= explorationFactor
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    m, s = divmod((time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print
    print "--- Execution time -> %d:%02d:%02d ---" % (h, m, s)
    # env.monitor.close()
