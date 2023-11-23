'''
This context-aware response manager interacts with users to provide personalized services during leisure time, serving as a companion system that converses on various topics. Inputs are received through text, while outputs are delivered via voice responses. This system is designed for verbally challenged children who experience loneliness. It serves as a companion by providing friendly responses to help children spend their leisure time effectively and enjoyably, relieving stress and promoting a positive mental state.
'''

# importing dependencies
import random as rn
import os
import transformers
import speech_recognition as sr
import pyttsx3
import pygame
import spacy
import warnings

# importing non-heavy libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# filter warnings
warnings.filterwarnings('ignore')

# loading some corpus
nlp = spacy.load('en_core_web_sm')

# data cleaning
punct = string.punctuation
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

# creating a function for data cleaning
def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
 
  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:    # Stopwords and punctuation removal
      cleaned_tokens.append(token)
  return cleaned_tokens

# tokenizer=text_data_cleaning, tokenization will be done according to this function
tfidf = TfidfVectorizer(tokenizer = text_data_cleaning, token_pattern = None)

# load the saved model from file using pickle
with open("D:/rf_model.pkl", 'rb') as f:
    rf = pickle.load(f)

# Greeting stage
def greeting():
    greetings = ["Hello", "Hi", "Hi there", "Hello there", "Hey", "Good day", "It's a pleasure to meet you", "Nice to see you", "Howdy"]
    random_response = rn.randint(0, len(greetings) - 1)
    return (greetings[random_response])

# Feedforward Stage
def feedforward():
    feedforwards = ["How do you do ?", "How are you ?", "How it's going ?"]
    random_feedforward = rn.randint(0, len(feedforwards) - 1)
    return (feedforwards[random_feedforward])

def about_MIRob():
    text = "I am Moratuwa Intelligent Robot, shortly known as MIRob. I am a joint product of the many research students of Intelligent Service Robotics group of University of Moratuwa. You can ask me anything"
    return (text)

def nice_hear():
    return ("It is nice to hear")   

def text_to_speech(text):
    # Convert the text to speech
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    # say anything - talk something
    print ("Prompt Something .....\n")
    transcript = input("prompt here: ")
    return (transcript) 

def about_me():
     text_to_speech(about_MIRob)
     chatbot_active()

def chatbot_active():
    #initialize pretrained model
    model = transformers.pipeline("conversational", model="facebook/blenderbot_small-90M")

    ## conversation runner code
    user = speech_to_text().strip().lower()

    classes = rf.predict([user])
    if classes[0] == "entertainment":
        keywords = user.split()
        #print (keywords)
        if "music" in keywords:
            music_player()
        elif "musics" in keywords:
            music_player()
        elif "joke" in keywords:
            tell_joke()
        elif "jokes" in keywords:
            tell_joke()
        elif "poem" in keywords:
            tell_poem()
        elif "poems" in keywords:
            tell_poem()

    else:
            
        # generate and print response
        response = str(model(transformers.Conversation(user)))
        #print(response[response.find("bot >> ")+6:].strip())
        text_to_speech(response[response.find("bot >> ")+6:].strip())
        #text_to_speech("what else do you want to ask me")
    return (user)

def play_music(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()



def music_player():
    # Path to your mp3 file
    music_file = ["D:/Travelling Soldier1.mp3", "D:/Bodhai Kaname.mp3", "D:/Kaattumalli.mp3","D:/Neenda Malare.mp3", "D:/Aaoromale.mp3", "D:/Panikaatrey.mp3"]
    random_response = rn.randint(0, len(music_file) - 1)
    play_music(music_file[random_response])
    while pygame.mixer.music.get_busy():
        continue
    text_to_speech("Hope you like it")
    print ("ready to ask!!")
    text_to_speech("What else do you want")
    # going back to main chatbot
    chatbot_active()

def tell_joke():
    joke_library = ["Why was the math book sad? Because it had too many problems", "What did one ocean say to the other ocean? Nothing, they just waved", "What do you call a dinosaur with an extensive vocabulary? A thesaurus", "What kind of tree fits in your hand? A palm tree", "Why do birds fly south in the winter? It's too far to walk", "What do you call a bee that comes from America? USB"]
    random_response = rn.randint(0, len(joke_library) - 1)
    # saying joke
    text_to_speech(joke_library[random_response])
    text_to_speech("That's a good joke right")
    text_to_speech("What else do you want")
    # going back to main chatbot
    chatbot_active()

def tell_poem():
    poem_library = ["Twinkle, twinkle, little star, How I wonder what you are! Up above the world so high, Like a diamond in the sky. Twinkle, twinkle, little star, How I wonder what you are!", "Rain, rain, go away, Come again another day. Little children want to play, Rain, rain, go away", "In the morning light, the world is fresh and new, a day full of promise, a chance to ask new."]
    random_response = rn.randint(0, len(poem_library) - 1)
    # saying poem
    text_to_speech(poem_library[random_response])
    text_to_speech("This is wonderful right")
    text_to_speech("What else do you want")
    # going back to main chatbot
    chatbot_active()



# System interacts with the user
#####################################
text_to_speech(greeting())
text_to_speech(feedforward())
query = speech_to_text()
text_to_speech(nice_hear())
text_to_speech(about_MIRob())
chatbot_active()
status = chatbot_active()
while status != 'bye':
    chatbot_active()
    status = chatbot_active()

# end of chat
#say("Good bye!")
text_to_speech("Good bye")   

#####################################

