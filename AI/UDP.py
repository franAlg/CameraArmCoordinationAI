import socket
import random
import numpy as np



class UDP:

    def __init__(self):
        self.UDP_IP = "127.0.0.1"
        self.CLIENT_PORT = 11000
        self.SERVER_PORT = 9900
        print "Initializing UDP server"
        self.sock = socket.socket(socket.AF_INET, # Internet
                             socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.UDP_IP, self.SERVER_PORT))

        self.lastAlfa = 0
        self.lastBeta = 0
        self.lastGamma = 0
        self.decay_rate = -1
        print "Connection established"

    def newEpisode(self):
        new = str(1)

        byteNew = new.encode()
        self.sock.sendto(byteNew, (self.UDP_IP, self.CLIENT_PORT))

    def noEpisode(self):
        no = str(0)

        byteNo = no.encode()
        self.sock.sendto(byteNo, (self.UDP_IP, self.CLIENT_PORT))

    def newObservation(self):

        data, addr = self.sock.recvfrom(4)
        #print "string recibido: ", data


        camX = int(float(data.replace(',','.')))
        # print "camX: ", camX
        data, addr = self.sock.recvfrom(4)
        camY = int(float(data.replace(',','.')))
        # print "camY: ", camY
        data, addr = self.sock.recvfrom(4)
        camZ = int(float(data.replace(',','.')))
        # print "camZ: ", camZ


        data, addr = self.sock.recvfrom(4)
        armX = int(float(data.replace(',','.')))
        # print "armX: ", armX
        data, addr = self.sock.recvfrom(4)
        armY = int(float(data.replace(',','.')))
        # print "armY: ", armY
        data, addr = self.sock.recvfrom(4)
        armZ = int(float(data.replace(',','.')))
        # print "armZ: ", armZ

        deltaX = armX - camX
        deltaY = armY - camY
        deltaZ = armZ - camZ

        delta = np.array([deltaX, deltaY, deltaZ])

        return delta

    def sendAction(self, action):

        anguloAlfa = repr(action[0])
        anguloBeta = repr(action[1])
        anguloGamma = repr(action[2])

        byteAlfa = anguloAlfa.encode()
        byteBeta = anguloBeta.encode()
        byteGamma = anguloGamma.encode();
        self.sock.sendto(byteAlfa, (self.UDP_IP, self.CLIENT_PORT))
        self.sock.sendto(byteBeta, (self.UDP_IP, self.CLIENT_PORT))
        self.sock.sendto(byteGamma, (self.UDP_IP, self.CLIENT_PORT))

        data, addr = self.sock.recvfrom(4)
        alfa = int(float(data.replace(',','.')))
        # print "deltaAlfa: ", alfa
        data, addr = self.sock.recvfrom(4)
        beta = int(float(data.replace(',','.')))
        # print "deltaBeta: ", beta
        data, addr = self.sock.recvfrom(4)
        gamma = int(float(data.replace(',','.')))
        # print "deltaGamma: ", gamma

        delta = np.array([alfa, beta, gamma])

        if alfa == 0 and beta == 0 and gamma == 0:
            done = True
        else :
            done = False

        self.lastAlfa = alfa
        self.lastBeta = beta
        self.lastGamma = gamma

        #devolver newObservation, reward, done
        return delta, done
