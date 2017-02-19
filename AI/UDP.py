import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 9900

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print "string recibido: ", data
    camX = float(data)
    print "dato recibido: ", camX
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    camY = float(data)
    print "dato recibido: ", camY
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    camZ = float(data)
    print "dato recibido: ", camZ

    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armX = float(data)
    print "dato recibido: ", armX
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armY = float(data)
    print "dato recibido: ", armY
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    armZ = float(data)
    print "dato recibido: ", armZ

    anguloAlfa = "20.0"
    anguloBeta = "30.0"
    anguloGamma = "40.0"

    sock.sendto(anguloAlfa, (UDP_IP, UDP_PORT))
    sock.sendto(anguloBeta, (UDP_IP, UDP_PORT))
    sock.sendto(anguloGamma, (UDP_IP, UDP_PORT))
