# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:46:12 2022

@author: iyngk
"""
print("hi")

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


data_path = "knowledge-neurons/data/PARAREL/data_all.json"
data_store = "data/data_all_allbags.json"
#enc = chardet.detect(open(data_path,"rb").read())["encoding"]
debug = 10000


#%% Create dataset
#Create and store dictionary with keys as relationships (e.g. P101)
#and values as corresponding relational facts

#json files seem appropriate to store text

with open(data_path,"r",encoding="utf-8") as f:
	eval_bag_list_all = json.load(f) #load contents from json file
	
	eval_bag_list_perrel = {}
	
	for bag_idx, eval_bag in enumerate(eval_bag_list_all):
		bag_rel = eval_bag[0][2].split("(")[0]
		if bag_rel not in eval_bag_list_perrel:
			eval_bag_list_perrel[bag_rel] = []
		
		print(len(eval_bag_list_perrel[bag_rel]))
		if len(eval_bag_list_perrel[bag_rel]) >= debug:
			continue
		eval_bag_list_perrel[bag_rel].append(eval_bag)
		
	with open(data_store,"w") as dstore:
		json.dump(eval_bag_list_perrel,dstore,indent=2)
		
#%% Auxilliary functions

#Get pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def example2feature(example, max_seq_length,tokenizer):
	"""
	Take a training example (h,t,r) and return input features
	
	tokens - words, subwords, letters
	token ids - Corresponding id in vocab. What actually goes into encoder
	"""

	features = []
	tokenslist = []
	
	ori_tokens = tokenizer.tokenize(example[0]) #Tokenize input sequence, list of strings
	
	if len(ori_tokens) > max_seq_length - 2:
		ori_tokens = ori_tokens[:max_seq_length-2] #Truncate token sequence if need be
		
		
	#BERT encoder needs segment ids too
	tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
	base_tokens = ["[UNK]"] + ["[UNK]"]*len(ori_tokens) + ["[UNK]"]
	segment_ids = [0]  * len(tokens)
	
	
	#Generate id and attention mask
	input_ids = tokenizer.convert_tokens_to_ids(tokens) #THIS is what is actually fed into encoder
	baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
	input_mask = [1] * len(input_ids)
	
	#PADDING so all input sequences of same length
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	baseline_ids += padding
	segment_ids += padding
	input_mask += padding #BINARY mask indicating proper tokens and padding tokens
	 
	assert len(baseline_ids) == max_seq_length
    
	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length    
	assert len(segment_ids) == max_seq_length
	
	features = {
		"input_ids":input_ids,
		"input_mask":input_mask,
		"segment_ids":segment_ids,
		"baseline_ids":baseline_ids}
	
	tokens_info = {
		"tokens":tokens,
		"relation":example[2],
		"gold_obj":example[1],
		"pred_obj":None}
	
	return features, tokens_info

def scaled_input(emb, batch_size, num_batch):
	#emb: (1,ffn_size)
	
	baseline = torch.zeros_like(emb) #(1,ffn_size)
	
	num_points = batch_size * num_batch
	step = (emb-baseline) / num_points #(1,ffn_size)
	
	res = torch.cat([torch.add(baseline,step*i) for i in range(num_points)]) #(num_points,ffn_size)
	
	return res, step[0]

def convert_to_triplet_ig(ig_list):
	 
	ig_triplet = []
	ig = np.array(ig_list) #(12,3072) igs across layers
	max_ig = ig.max()
	for i in range(ig.shape[0]):
		for j in range(ig.shape[1]):
			if ig[i][j] >= max_ig * 0.1: #If ig[i][j] exceeds value
				ig_triplet.append(i,j,ig[i][j]) #neuron location and ig value
	return ig_triplet
	

#%% Get model + tokenizer + device

tokenizer = BertTokenizer.from_pretrained("bert-base-cased") #Pretrained BERT tokenizer

print("***CUDA.empty_cache()***") #Release memory in cache for new run
torch.cuda.empty_cache()
model = BertForMaskedLM.from_pretrained("bert-base-cased")

if not torch.cuda.is_available():
	device = torch.device("cpu")
	n_gpu = 0
else:
	device = torch.device("cuda:1")
	n_gpu = 1

model.to(device)

if n_gpu>1:
	model = torch.nn.DataParallel(model) #Parallel processing across gpus
	
model.eval()
#%% Setup


#Get data
with open (data_store, "r",encoding="utf-8") as fl:
	eval_bag_list_rels = json.load(fl)

#Params
max_seq_length = 128
rel_to_eval = "P101"
get_pred = True
get_ig_pred = True
get_ig_gold = True
get_base = True #Get baseline attribution (just weights)

batch_size = 20
num_batch = 1



#%% Main loop	
for relation, eval_bag_list in eval_bag_list_rels.items():
	if relation != rel_to_eval:
		continue
	
	start_time = time.time()
	
	for bag_idx, eval_bag in enumerate(eval_bag_list):
		
		#Store results
		res_dict_bag = []
		
		for eval_example in eval_bag:
			eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
			
			#Convert features into long type tensors
			baseline_ids, input_ids, input_mask, segment_ids = eval_features["baseline_ids"], eval_features["input_ids"],eval_features["input_mask"],eval_features["segment_ids"]
			baseline_ids = torch.tensor(baseline_ids,dtype=torch.long).unsqueeze(0) #add dim
			input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
			input_mask = torch.tensor(input_mask,dtype=torch.long).unsqueeze(0)
			segment_ids = torch.tensor(segment_ids,dtype=torch.long).unsqueeze(0)
			
			#Determine which device tensor to be allocated to
			baseline_ids = baseline_ids.to(device)
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			
			input_len = int(input_mask[0].sum())
			
			#Record indices of "[MASK]" tokens
			tgt_pos = tokens_info["tokens"].index("[MASK]")
			
			res_dict = {
				"pred":[],
				"ig_pred":[],
				"ig_gold":[],
				"base":[]
				}
			
			if get_pred:
				_,logits = model(input_ids=input_ids,
					 attention_mask=input_mask, #Don't want to compute attention between proper token ids and padding tokens
					 token_type_ids=segment_ids,
					 tgt_pos = tgt_pos,
					 tgt_layer=0)
			


	
#%% Test ground(single example)

st_time = time.time()

rel_facts_list = eval_bag_list_rels[rel_to_eval] #ALL rel facts corresponding to particular rel
rel_fact_ht = rel_facts_list[0] #All rel facts for particular (h,t) pair)
example = rel_fact_ht[0] #ONE rel fact 


#Store results
res_dict_bag = []


eval_features, tokens_info = example2feature(example, max_seq_length, tokenizer)

#Convert features into long type tensors
baseline_ids, input_ids, input_mask, segment_ids = eval_features["baseline_ids"], eval_features["input_ids"],eval_features["input_mask"],eval_features["segment_ids"]
baseline_ids = torch.tensor(baseline_ids,dtype=torch.long).unsqueeze(0) #add dim (corresponding to batch size?)
input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
input_mask = torch.tensor(input_mask,dtype=torch.long).unsqueeze(0)
segment_ids = torch.tensor(segment_ids,dtype=torch.long).unsqueeze(0)

#Determine which device tensor to be allocated to
baseline_ids = baseline_ids.to(device)
input_ids = input_ids.to(device)
input_mask = input_mask.to(device)
segment_ids = segment_ids.to(device)

input_len = int(input_mask[0].sum())

#Record indices of "[MASK]" tokens - the 'target' positions
tgt_pos = tokens_info["tokens"].index("[MASK]") #Masked tokens indices

res_dict = {
	"pred":[],
	"ig_pred":[],
	"ig_gold":[],
	"base":[]
	}

if get_pred:
	weights,logits = model(input_ids=input_ids,
		 attention_mask=input_mask, #Don't want to compute attention between proper token ids and padding tokens
		 token_type_ids=segment_ids,
		 tgt_pos = tgt_pos,
		 tgt_layer=0) #Think first output are WEIGHTS corresponding to tgt position
	base_pred_prob = F.softmax(logits,dim=1)
	res_dict["pred"].append(base_pred_prob.tolist())
	
#tgt - target
for tgt_layer in range(model.bert.config.num_hidden_layers):
	ffn_weights, logits = model(input_ids = input_ids,
							 attention_mask = input_mask,
							 token_type_ids = segment_ids,
							 tgt_pos = tgt_pos,
							 tgt_layer = tgt_layer)

	pred_label = int(torch.argmax(logits[0,:])) #Find predicted token INDEX
	gold_label = tokenizer.convert_tokens_to_ids(tokens_info["gold_obj"]) #actual index
	tokens_info["pred_obj"] = tokenizer.convert_ids_to_tokens(pred_label)
	scaled_weights, weights_step = scaled_input(ffn_weights,batch_size,num_batch)
	scaled_weights.requires_grad_(True) #Required for backprop
	
	
	#For a particular layer, l
	#How much can neuron i be attributable to predicted label
	#Output is of dimension #(1,d_m) where d_m is dimension of hidden state
	if get_ig_pred:
		ig_pred = None #instantiate
		for batch_idx in range(num_batch):
			batch_weights = scaled_weights[batch_idx*batch_size:(batch_idx + 1)*batch_size]
			_,grad = model(input_ids = input_ids,
				  attention_mask = input_mask,
				  token_type_ids = segment_ids,
				  tgt_pos = tgt_pos,
				  tgt_layer = tgt_layer, 
				  tmp_score = batch_weights, #
				  tgt_label = pred_label)
			grad = grad.sum(dim=0)  
			ig_pred = grad if ig_pred is None else torch.add(ig_pred,grad) #(ffn_size)
		ig_pred = ig_pred * weights_step #Multiply by differential
		res_dict["ig_pred"].append(ig_pred.tolist())
	
	
	#for a particular layer, l
	#How much can neuron i be attributable to predicting the gold label?
	#Dimension is of #(1,d_m) again
	if get_ig_gold:
		ig_gold = None
		#number of steps in Riemannian approx
		for batch_idx in range(num_batch):
			batch_weights = scaled_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]
			_,grad = model(input_ids = input_ids,
				  attention_mask = input_mask,
				  token_type_ids = segment_ids,#BERT needs segment ids, but here all are 0 (no NSP)
				  tgt_pos = tgt_pos,
				  tmp_score = batch_weights,
				  tgt_label = gold_label)
			grad = grad.sum(dim=0) 
			ig_gold = grad if ig_gold is None else torch.add(ig_gold,grad)
		ig_gold = ig_gold * weights_step
		res_dict["ig_gold"].append(ig_gold.tolist())
		
	if get_base:
		res_dict["base"].append(ffn_weights.squeeze().tolist()) #baseline attribution method
		
		

#Only store ig values greater than 0.1*max_ig value for particular relational fact
res_dict["ig_gold"] = convert_to_triplet_ig(res_dict["ig_gold"])

res_dict_bag.append([tokens_info,res_dict])

#THEN WRITE TO JSON FILE
#AND DONE! ig calculated for a single relational fact!
#Now do this for all facts :0

		
		
				  
			
		
		
		
		
end = time.time()

elapsed = end-st_time
	


print("hey");
#Looks like this is working!