import os

try:
    os.environ['OPENAI_API_KEY']='JL1qBgjTVlQ24Oo27RswT3BlbkFJFFJWBsxLBNnZgR64qc8G'
    print(os.environ['OPENAI_API_KEY'])
except:
    print('Error setting API key')


print(os.environ)