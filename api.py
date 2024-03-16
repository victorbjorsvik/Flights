import os

try:
    os.environ['OPENAI_API_KEY']='***REMOVED***'
    print(os.environ['OPENAI_API_KEY'])
except:
    print('Error setting API key')


print(os.environ)