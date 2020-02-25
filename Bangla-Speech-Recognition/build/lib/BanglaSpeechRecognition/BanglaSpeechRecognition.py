# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:12:59 2020

@author: Kowsher
"""

import speech_recognition as sr  

# get audio from the microphone
def STT_Bangla():                                                                       
    r = sr.Recognizer()                                                                                   
    with sr.Microphone() as source:                                                                                                                                                        
        audio = r.listen(source)   
    
    try:
         string = r.recognize_google(audio, language = 'bn')
    except sr.UnknownValueError:
         string = "Could not understand audio"
    except sr.RequestError as e:
         string = "Could not request results; {0}".format(e)
    return string
def version():
    st = "1.00"
    return st
def developer():
    st = "The BSTT is developed by 'Md. Kowsher'"
    return st
