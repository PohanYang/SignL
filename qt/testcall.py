import sys
import fbchat
import chating

if __name__ =='__main__':
	client = fbchat.Client("scure.le.1", sys.argv[1])
	friends = client.getUsers("Pohan Yang")
	friend = friends[0]
	chating.fb_buffer(client, friend)
	print "a"
