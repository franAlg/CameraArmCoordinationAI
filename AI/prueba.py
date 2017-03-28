import UDP

name = UDP.UDP()

name.newEpisode()
observation = name.newObservation()

#print "-------------------------------", observation

i = 0

while i < 500 :
    if i != 0 :
        if i == 5 :
            name.newEpisode()
        else :
            name.noEpisode()
    action = [i,i,i]
    name.sendAction(action)
    i += 1
