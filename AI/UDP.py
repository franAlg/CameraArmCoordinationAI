import socket
import random
import numpy as np



class UDP:

    def __init__(self):
        self.UDP_IP = "127.0.0.1"
        self.CLIENT_PORT = 11000
        self.SERVER_PORT = 9900
        print "Inciando servidor UDP"
        self.sock = socket.socket(socket.AF_INET, # Internet
                             socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.UDP_IP, self.SERVER_PORT))

        self.lastDelta = np.array([0,0,0])
        self.lastDist = 0
        self.decay_rate = -1
        print "conexion establecida"

    def newEpisode(self):
        new = str(1)

        byteNew = new.encode()
        self.sock.sendto(byteNew, (self.UDP_IP, self.CLIENT_PORT))

        print
        #print "nuevo episodio"

    def noEpisode(self):
        no = str(0)

        byteNo = no.encode()
        self.sock.sendto(byteNo, (self.UDP_IP, self.CLIENT_PORT))

    def newObservation(self):
        #print "esperando a recibir datos"
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        #print "string recibido: ", data

        print

        camX = int(float(data.replace(',','.')))
        # print "camX: ", camX
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        camY = int(float(data.replace(',','.')))
        # print "camY: ", camY
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        camZ = int(float(data.replace(',','.')))
        # print "camZ: ", camZ

        print

        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        armX = int(float(data.replace(',','.')))
        # print "armX: ", armX
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        armY = int(float(data.replace(',','.')))
        # print "armY: ", armY
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        armZ = int(float(data.replace(',','.')))
        # print "armZ: ", armZ

        # print "todos los datos recibidos"

        deltaX = armX - camX
        deltaY = armY - camY
        deltaZ = armZ - camZ

        delta = np.array([deltaX, deltaY, deltaZ])

        return delta

    def sendAction(self, action):

        # print

        # print "alfa: ", action[0]
        # print "beta: ", action[1]
        # print "gamma: ", action[2]

        anguloAlfa = repr(action[0])
        anguloBeta = repr(action[1])
        anguloGamma = repr(action[2])

        # print

        # print "enviando respuesta"

        byteAlfa = anguloAlfa.encode()
        byteBeta = anguloBeta.encode()
        byteGamma = anguloGamma.encode();
        self.sock.sendto(byteAlfa, (self.UDP_IP, self.CLIENT_PORT))
        self.sock.sendto(byteBeta, (self.UDP_IP, self.CLIENT_PORT))
        self.sock.sendto(byteGamma, (self.UDP_IP, self.CLIENT_PORT))

        # print "respuesta enviada"
        # print "esperando resultado"
        # print

        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        alfa = int(float(data.replace(',','.')))
        # print "deltaAlfa: ", alfa
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        beta = int(float(data.replace(',','.')))
        # print "deltaBeta: ", beta
        data, addr = self.sock.recvfrom(4) # buffer size is 1024 bytes
        gamma = int(float(data.replace(',','.')))
        # print "deltaGamma: ", gamma

        # print
        # print "-------------------------------"

        delta = np.array([alfa,beta,gamma])


        #ajustar tanto para arriba como por abajo
        if alfa <= 1 and alfa >= -1 and beta <= 1 and beta >= -1 and gamma <= 1 and gamma >= -1 :
            done = True

            reward = 1.0

            # print "-------------------------------"
            # print "last distance: ", self.lastDist
            # print "actual distance: ", np.linalg.norm(delta-self.lastDelta)
            # print "reward : ", reward
            # print "-------------------------------"

        else :
            done = False

            distance = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
            reward = np.exp(self.decay_rate * distance)

            # print "-------------------------------"
            # print "last distance: ", self.lastDist
            # print "actual distance: ", np.linalg.norm(delta-self.lastDelta)
            # print "reward : ", reward
            # print "-------------------------------"

            self.lastDist = np.linalg.norm(delta-self.lastDelta)

        #devolver newObservation, reward, done
        return delta, reward, done
