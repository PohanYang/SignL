import fbchat
import pickle
import sys

def fb_buffer(client):
	while(1):
		chofid = client.getUsers("Pohan Yang")
		chofid = chofid[0]
		fbm = ''
		last_messages = client.getThreadInfo(chofid.uid, last_n=7)
		last_messages.reverse()
		for message in last_messages:
			if str(message.author.split(':')[1])==str(chofid.uid):
			    fbm = fbm+"Pohan Yang: "+message.body + "\n"
			else:
			    fbm = fbm+"Me: "+message.body + "\n"
		with open("fbchat.p", "wb") as fp:
			pickle.dump(fbm, fp)

if __name__ == '__main__':
	client = fbchat.Client("scure.le.1", sys.argv[1])
	fb_buffer(client)
