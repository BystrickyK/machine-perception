#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:30:06 2020

@author: christoph
"""


class EncryptoDecryptor():
    
    # Class attribute
    alphabet = 'abcdefghijklmnopqrstuvwxyz ' 
    # alphabet = 'abcdefghijklmnopqrstuvwxyz !@#$' 
    
    def __init__(self, key):
        self.key = key
        
    @property
    def key(self):
        return self._key
    
    # Setting the key property also sets ciphertable/deciphertable attributes
    @key.setter
    def key(self, key):
        
        alphabet = EncryptoDecryptor.alphabet
        # Check for key validity & set self.key
        if key > len(alphabet) or key < 0:
            self._key = key%len(alphabet)
            print('Invalid key => key instead initialized as mod(key,{len_alphabet})== {new_key}'.format(
            len_alphabet=len(alphabet), new_key=key))
        else: 
            self._key = key
            
        cipherbet = alphabet[self._key:] + alphabet[:self._key]
        # Dict comprehension
        self.ciphertable = {plain:cipher for plain,cipher in zip(alphabet,cipherbet)}
        # Dict inversion
        self.deciphertable = {cipher:plain for plain,cipher in self.ciphertable.items()}
    
    def encrypt(self, plaintext):
        # Chars not present in the alphabet default to #
        return ''.join([self.ciphertable.get(char, '#') for char in plaintext])

    def decrypt(self, ciphertext):
        return ''.join([self.deciphertable.get(char, '#') for char in ciphertext])
    

message = "do not eat all my oranges !!##$$test@@"

key = 1
encrypto_decryptor = EncryptoDecryptor(key)

print("Key:\t\t\t" + str(encrypto_decryptor.key))
print("Message:\t\t" + message)
print("Ciphertext:\t\t" + (ciphertext := encrypto_decryptor.encrypt(message)))
print("Plaintext:\t\t" + (plaintext := encrypto_decryptor.decrypt(ciphertext)))

# Demonstrate that changing the key alone also changes the cipher output
encrypto_decryptor.key = 2
print("Key:\t\t\t" + str(encrypto_decryptor.key))
print("Message:\t\t" + message)
print("Ciphertext:\t\t" + (ciphertext := encrypto_decryptor.encrypt(message)))
print("Plaintext:\t\t" + (plaintext := encrypto_decryptor.decrypt(ciphertext)))

# =============================================================================
# (a := 2) ----> Assigns 2 to variable 'a' and returns the assigned value (2)
# Known as the 'walrus operator'
# =============================================================================
    
    
    
    
    
    