# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:01:16 2019

@author: ASUS-PC
"""
import tweepy
 
# consumer keys and access tokens, used for OAuth
consumer_key = 'YOUR-CONSUMER-KEY'
consumer_secret = 'YOUR-CONSUMER-SECRET'
access_token = 'YOUR-ACCESS-TOKEN'
access_token_secret = 'YOUR-ACCESS-TOKEN-SECRET'
 
# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# creation of the actual interface, using authentication
api = tweepy.API(auth)