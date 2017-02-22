import socket
import random

UDP_IP = "127.0.0.1"
CLIENT_PORT = 11000
SERVER_PORT = 9900

print "Inciando servidor UDP"
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, SERVER_PORT))
print "conexion establecida"

while True:
    print "esperando a recibir datos"
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print "string recibido: ", data

    print

    camX = float(data.replace(',','.'))
    print "camX: ", camX
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    camY = float(data.replace(',','.'))
    print "camY: ", camY
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    camZ = float(data.replace(',','.'))
    print "camZ: ", camZ

    print

    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armX = float(data.replace(',','.'))
    print "armX: ", armX
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armY = float(data.replace(',','.'))
    print "armY: ", armY
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armZ = float(data.replace(',','.'))
    print "armZ: ", armZ

    print "todos los datos recibidos"

    print

    alfa = random.uniform(0, 100)
    print "alfa: ", alfa
    beta = random.uniform(0, 100)
    print "beta: ", beta
    gamma = random.uniform(0, 100)
    print "gamma: ", gamma

    anguloAlfa = repr(alfa)
    anguloBeta = repr(beta)
    anguloGamma = repr(gamma)

    print

    print "enviando respuesta"

    byteAlfa = anguloAlfa.encode()
    byteBeta = anguloBeta.encode()
    byteGamma = anguloGamma.encode();
    sock.sendto(byteAlfa, (UDP_IP, CLIENT_PORT))
    sock.sendto(byteBeta, (UDP_IP, CLIENT_PORT))
    sock.sendto(byteGamma, (UDP_IP, CLIENT_PORT))

    print "respuesta enviada"

    print

    print "esperando resultado"

    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    distancia = float(data.replace(',','.'))

    print "distancia recibida ", distancia

    print "-------------------------------"
