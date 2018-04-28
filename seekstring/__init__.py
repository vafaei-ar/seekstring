modules = ['data_provider','networks','utils']

for module in modules:
	exec 'from '+module+' import *'

