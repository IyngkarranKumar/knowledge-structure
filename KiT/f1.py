# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 12:28:07 2022

@author: iyngk
"""

#%% IMPORT
import argparse
import json,jsonlines
import chardet
import time
import random; random.seed(4) #For reproducibility
import sys

import numpy as np; np.random.seed(4)
import torch; torch.manual_seed(4)
if torch.cuda.is_available:
	torch.cuda.manual_seed_all(4)
#Hugging face transformer library
import transformers
#The tokenizer used to train Bert
from transformers import BertTokenizer
#Bert model purely for masked language modelling (no sentence ordering)
from custom_bert import BertForMaskedLM

import torch.nn.functional as F


#%% Data + big boys to load

dataset_path = "data/processed_data.json"

with open(dataset_path,"r",encoding="utf-8") as fo:
	dataset = json.load(fo)
	
relations = list(dataset.keys())
attribution_data_path = "data/attribution_data.json"
	
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForMaskedLM.from_pretrained("bert-base-cased")



if torch.cuda.is_available():
	dev = "cuda:0"
else:
	dev = "cpu"
device = torch.device(dev)
model.to(device) #Optimise model for particular runtime device

print("***CUDA.empty_cache()***") #Release memory in cache for new run
torch.cuda.empty_cache()

#%% Auxilliary function 

def convert_seq_to_feature(fact,tokenizer,max_seq_length):
	"""
	Prepare fact for insertion into MaskedLM
	"""
	
	input_seq = fact[0]; target = fact[1]
	
	tokens = ["[CLS]"]+tokenizer.tokenize(input_seq)+["[SEP]"]
	
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	segment_ids = [0]*len(tokens)
	input_mask = [1]*len(tokens)
	
	#padding for max_seq_length
	padding = [0]*(max_seq_length - len(tokens))
	input_ids+=padding
	segment_ids+=padding
	input_mask+=padding
	

	
	feature_info = {
		"input_ids":input_ids,
		"segment_ids":segment_ids,
		"input_mask":input_mask}
	
	tokens_info = {
		"tokens":tokens,
		"target":target}
	
	return feature_info, tokens_info
	
def scale_weights(ffn_weights, num_steps):

    scaled_weights = []
    
    weight_step = ffn_weights / num_steps
    
    for i in range(1,num_steps+1):
        
        scaled_weights_i = weight_step * i
        scaled_weights.append(scaled_weights_i)
        
    scaled_weights = torch.cat(scaled_weights)
        

    #assert(torch.eq(ffn_weights,scaled_weights[-1]))
    
    return scaled_weights, weight_step


def ig_triplet(ig_list):
    
    ig_triplet = []
    ig = np.array(ig_list) #(12,3072) igs across layers
    max_ig = ig.max()
    for i in range(ig.shape[0]):
   		for j in range(ig.shape[1]):
   			if ig[i][j] >= max_ig * 0.1: #If ig[i][j] exceeds value
   				ig_triplet.append(i,j,ig[i][j]) #neuron location and ig value
    
    return ig_triplet
    
        

	



#%% Test

#PARAMS
num_steps = 20 #num steps in Riemann sum

fact = dataset[relations[0]][0][0]
feature = fact[0]; target = fact[1]

feature_info, tokens_info = convert_seq_to_feature(fact,tokenizer,128)
input_ids = feature_info["input_ids"]
segment_ids = feature_info["segment_ids"]
input_mask = feature_info["input_mask"]
tokens = tokens_info["tokens"]
target = tokens_info["target"]


#Convert to torch tensor - tokenizer returns list
input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
segment_ids = torch.tensor(segment_ids,dtype=torch.long).unsqueeze(0)
input_mask = torch.tensor(segment_ids,dtype=torch.long)


input_ids = input_ids.to(device)
segment_ids = segment_ids.to(device)
input_mask = input_mask.to(device)

target_position = tokens.index("[MASK]")


#Now iterate over all layers to find attribution for each neuron

results = {}
fact_results = {
    "ig_gold":[],
    "base":[]}

for layer in range(model.config.num_hidden_layers):
    
    #Return FFN hidden layer activations + logits for particular FFN layer, at particular position
    ffn_weights, logits = model(input_ids = input_ids,
                                attention_mask = input_mask,
                                token_type_ids = segment_ids,
                                tgt_pos = target_position,
                                tgt_layer = layer)
   
    #Scale weights to compute Riemann sum 
    scaled_weights, weight_step = scale_weights(ffn_weights,num_steps=20)
    scaled_weights.requires_grad_(True) #Record operations for backprop
    
    ig_gold = None #instantiate integrated gradients at gold label
    for step in range(num_steps):
        
        step_weights = scaled_weights[step]
        _,grad = model(input_ids = input_ids,
                       attention_mask = input_mask, 
                       token_type_ids = segment_ids,
                       tgt_pos = target_position, #get gradients for this token,
                       tmp_score = step_weights #temporary
                       )
        
        grad = grad.sum(dim=0)
        ig_gold = grad if ig_gold is None else torch.add(ig_gold,grad)
       
    #ig at gold label at layer l
    ig_gold = ig_gold * weight_step #multiply by differential
    fact_results["ig_gold"].append(ig_gold.tolist())
    
    base = ffn_weights.squeeze()
    fact_results["base"].append(base)
    
fact_results["ig_gold"] = ig_triplet(fact_results["ig_gold"]) # convert ig_gold data to triplet (more amenable to analysis)


results[fact] = fact_results


    

#%% Main loop

#Parameters
num_steps = 20 #num steps in Riemann sum

results_path = "data/results.json"

def main():

    for relation in relations:
        for hrt_triplets in dataset[relations]:
            for fact in hrt_triplets:

                feature = fact[0]; target = fact[1]

                feature_info, tokens_info = convert_seq_to_feature(fact,tokenizer,128)
                input_ids = feature_info["input_ids"]
                segment_ids = feature_info["segment_ids"]
                input_mask = feature_info["input_mask"]
                tokens = tokens_info["tokens"]
                target = tokens_info["target"]


                #Convert to torch tensor - tokenizer returns list
                input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids,dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(segment_ids,dtype=torch.long)


                input_ids = input_ids.to(device)
                segment_ids = segment_ids.to(device)
                input_mask = input_mask.to(device)

                target_position = tokens.index("[MASK]")


                #Now iterate over all layers to find attribution for each neuron

                results = {}
                fact_results = {
                    "ig_gold":[],
                    "base":[]}

                for layer in range(model.config.num_hidden_layers):
                    
                    #Return FFN hidden layer activations + logits for particular FFN layer, at particular position
                    ffn_weights, logits = model(input_ids = input_ids,
                                                attention_mask = input_mask,
                                                token_type_ids = segment_ids,
                                                tgt_pos = target_position,
                                                tgt_layer = layer)
                    
                    #Scale weights to compute Riemann sum 
                    scaled_weights, weight_step = scale_weights(ffn_weights,num_steps=20)
                    scaled_weights.requires_grad_(True) #Record operations for backprop
                    
                    ig_gold = None #instantiate integrated gradients at gold label
                    for step in range(num_steps):
                        
                        step_weights = scaled_weights[step]
                        _,grad = model(input_ids = input_ids,
                                       attention_mask = input_mask, 
                                       token_type_ids = segment_ids,
                                       tgt_pos = target_position, #get gradients for this token,
                                       tmp_score = step_weights #temporary
                                       )
                        
                        grad = grad.sum(dim=0)
                        ig_gold = grad if ig_gold is None else torch.add(ig_gold,grad)
                        
                        #ig at gold label at layer l
                        ig_gold = ig_gold * weight_step #multiply by differential
                        fact_results["ig_gold"].append(ig_triplet(ig_gold.tolist()))

                        base = ffn_weights.squeeze()
                        fact_results["base"].append(base)
                        
                        results[str(fact)] = fact_results
                        
                        with open(results_path,"a"):
                            json.dumps(fact_results)
                    


