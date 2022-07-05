
#%% Imports

from lib2to3.pgen2.tokenize import tokenize
from os import truncate
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import inspect
import logging
from timeit import default_timer


import torch
from torch.nn import functional as F
import transformers
from transformers import BertTokenizer, BertModel
import datasets


#%% Auxilliary functions
if 1: 

    def prefix2feature(prefix,tokenizer,max_length=128):

        input_data = tokenizer(prefix,padding="max_length",truncation=True,max_length=max_length)
        input_ids = torch.tensor(input_data["input_ids"],requires_grad=False).unsqueeze(0)
        attention_mask = torch.tensor(input_data["attention_mask"],requires_grad=False).unsqueeze(0)

        return input_ids, attention_mask

    def sentence2feature(sentence,tokenizer,max_length=128):
        
        word_list = sentence.split()
        n_words = len(word_list)

        for i in range(n_words):
            
            prefix = " ".join(word_list[:i+1])
            input_data = tokenizer(prefix,padding="max_length",truncation=True,max_length=max_length)
            
            if i==0:
                batch_input_ids = torch.tensor(input_data["input_ids"],requires_grad=False) #no backprop on this tensor - so don't store computational graph
                batch_attention_mask = torch.tensor(input_data["attention_mask"],requires_grad=False)

                batch_input_ids = batch_input_ids[None,:]
                batch_attention_mask = batch_attention_mask[None,:]

            else:
                input_ids = torch.tensor(input_data["input_ids"],requires_grad=False)[None,:] #no backprop on this tensor - so don't store computational graph
                attention_mask = torch.tensor(input_data["attention_mask"],requires_grad=False)[None,:]

                batch_input_ids = torch.cat((batch_input_ids,input_ids),dim=0)
                batch_attention_mask = torch.cat((batch_attention_mask,attention_mask),dim=0)

        return batch_input_ids,batch_attention_mask

    def calc_mean_prefix_embedding(sentence_embeddings):
        """
        Input: word embeddings for the n prefixes composing a sentence (n_prefixes, max_seq_length, emb_size)
        Output: Intermediate prefix embeddings for the n prefixes composing a sentence (n_prefixes, emb_size)
        """

        n_prefixes = sentence_embeddings.size(0)
        seq_length = sentence_embeddings.size(1)
        emb_size = sentence_embeddings.size(2)

        mean_prefix_embeddings = torch.zeros(size=(n_prefixes,emb_size),dtype=torch.float)

        for i in range(n_prefixes):
            prefix_embedding = sentence_embeddings[i]
            #We only want to average first i+1 rows (rest are PADDING)
            mean_prefix_embeddings[i] = torch.mean(prefix_embedding[0:i+1],dim=0)

        assert(mean_prefix_embeddings.size()==(n_prefixes, emb_size))

        return mean_prefix_embeddings


    def get_semanticity_score(bag_of_prefixes,tokenizer,
    n_common_toks=10, min_counts_perc=1.0,posn_weight=0.2):

        #Calculate repetition score
        input_ids = tokenizer(bag_of_prefixes)["input_ids"]
        for idx,tokenized_prefix in enumerate(input_ids):

            if idx==0:
                list_of_tokens = tokenized_prefix
            else:
                list_of_tokens = list_of_tokens + tokenized_prefix

        #Probably can write this better
        list_of_tokens = [i for i in list_of_tokens if i > 1116] #take any token after ##s
        uniqs,counts = np.unique(list_of_tokens,return_counts=True)
        repeated_tokens = uniqs[counts>=min_counts_perc*len(bag_of_prefixes)]
        rep_tok_counts = counts[uniqs==repeated_tokens]

        S_sem = rep_tok_counts / len(bag_of_prefixes)

        if len(repeated_tokens) > 1:
            print("Warning: Multiple repeated patterns identfied")
            print(repeated_tokens)

       #Calculate position score
        for idx,token in enumerate(repeated_tokens):
            posns = []
            for tokenized_prefix in input_ids:
                if token not in tokenized_prefix:
                    continue
                len_prefix = len(tokenized_prefix)
                idx_token = tokenized_prefix.index(token) #ASSUMES TOKEN APPEARS JUST ONCE

                if idx_token < int(len_prefix/3):
                    posns.append(0)
                elif int(len_prefix/3) <= idx_token < 2*int(len_prefix/3):
                    posns.append(1)
                else:
                    posns.append(2)

            _,counts = np.unique(posns,return_counts=True)
            H = entropy(counts,base=2) #Shannon entropy
            H_max = entropy([1,1,1],base=2)
            S_sem_posn = posn_weight * (1-(H/H_max))

            S_sem[idx] = S_sem[idx] + S_sem_posn #Add position semanticity score
        
        return S_sem

    def get_object_memory_footprint():
        #https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter

        # These are the usual ipython objects, including this one you are creating
        ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

        # Get a sorted list of the objects and their sizes
        sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
            

#%% Dataset
if 1: 
    dataset = datasets.load_dataset("wikitext","wikitext-103-v1",split="train")
    dataset_text = dataset["text"]

#%% Set device
if 1: 
    if torch.cuda.is_available():
        dev = "cuda"
    else: 
        dev = "cpu"
    device = torch.device(dev)
    print("Device: {}".format(device))
    print("***CUDA empty cache***")
    torch.cuda.empty_cache()




#%% Tokenizer and model
if 1: 
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")

    model = model.to(device)



    def get_intermediate_embeddings(name):
        def hook_func(module,input,output):
            embeddings[name] = input[0].detach()
        return hook_func
 
    
    #register hook
    for i in range(model.config.num_hidden_layers):
        model.encoder.layer[2].intermediate.register_forward_hook(get_intermediate_embeddings("layer"+str(i)))


    #named parameters dictionary
    named_params = dict(model.named_parameters())

#%%

if 0: 
    """
    To do: Find top-N memory coefficients for one neuron
        - Find way to send textual data in one prefix at a time
        - Find efficient method of storing top-N memory coefficients
    """
    #Intermediate activations store
    memory_coeffs = {}
    #Params
    min_sentence_length = 50

    neuron_idxs = (5,5)
    assert(0 < neuron_idxs[0] <= model.config.num_hidden_layers)
    assert(0 < neuron_idxs[1] <= model.config.intermediate_size)



    #forward hook
    #EXAMPLE: model.linear.register_forward_pre_hook(hook)

    


    for txt in sample_text:
        
        #Skip titles - only want paragraphs
        if len(txt)<50:
            continue 

        for sentence in txt.split("."):

            
            sentence = "The capital of Germany is Berlin"


            activations = {}
            batch_input_ids, batch_attention_mask = sentence2feature(sentence,tokenizer)
            batch_input_ids = batch_input_ids.to(device);
            batch_attention_mask = batch_attention_mask.to(device)


            outputs = model(input_ids = batch_input_ids,
            attention_mask = batch_attention_mask
            )

            for i in range(model.config.num_hidden_layers):

                param_name = "encoder.layer."+str(i)+".intermediate.dense.weight"
                KEYS = named_params[param_name].detach() #don't record operations

                sentence_embedding = activations["layer"+str(i)]
                mean_prefix_embedding = calc_mean_prefix_embeddings(sentence_embedding).to(device)

                memory_coeffs = F.relu((torch.matmul(mean_prefix_embedding,torch.transpose(KEYS,0,1)))) #As given in Geva et al.





                print("out")
                sys.exit()

   
#%%

if 1:
#Create storage data structure
    N = 5
    topN_prefixes = {}

    for layer_idx in range(model.config.num_hidden_layers):
        topN_prefixes[layer_idx] = [[] for i in range(model.config.intermediate_size)]

    #begin loop
    n_prefixes_analysed = 0
    times = []

    sample_text = dataset_text[:20]
   
    #get paragraph from text
    for txt in sample_text:

        #Skip titles - only want paragraphs
        if len(txt)<50:
                continue 

        #get sentence
        for sentence in txt.split("."):


            word_list = sentence.split()

            prefixes = [" ".join(word_list[:i+1]) for i in range(len(word_list))]


            #get prefix
            
            for prefix in prefixes:
                start = default_timer()
                print("Analysing prefix: {}".format(prefix))
                print("Number of prefixes analysed: {}".format(n_prefixes_analysed))

                embeddings = {} #To store prefix WORD embeddings
                memory_coefficients = {} #To store prefix memory coefficients for each key

                input_ids,attention_mask = prefix2feature(prefix,tokenizer)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                _ = model(input_ids = input_ids,
                attention_mask = attention_mask)

                #get model layer
                for layer_idx in range(model.config.num_hidden_layers):

                    prefix_layer_embeddings = embeddings["layer"+str(layer_idx)] #word embeddings for this prefix in layer l, 
                    #(max_seq_length,emb_size)

                    mean_prefix_embedding = calc_mean_prefix_embedding(prefix_layer_embeddings) 
                    mean_prefix_embedding = mean_prefix_embedding.to(device)
                    #(1,emb_size)

                    param_matrix_name = "encoder.layer."+str(layer_idx)+".intermediate.dense.weight"
                    KEYS = named_params[param_matrix_name].detach()

                    #memory coefficients for (prefix,layer) 
                    prefix_layer_mcs = F.relu((torch.matmul(mean_prefix_embedding,torch.transpose(KEYS,0,1)))) #As given in Geva et al.
                    torch.round(prefix_layer_mcs,decimals=5)

                    #iterate over memory coefficients (i.e keys)
                    for i,mc in enumerate(prefix_layer_mcs.tolist()[0]):

                        #Get key topN prefix data structure
                        key_topN_data = topN_prefixes[layer_idx][i]

                        if len(key_topN_data) < N:
                            key_topN_data.append((mc,prefix))

                        else:

                            #get memory coefficients
                            topN_mcs = [k for (k,j) in key_topN_data] 
                            
                            #assumes N UNIQUE memory coefficients
                            if mc > min(topN_mcs):
                                idx_to_replace = np.argmin(topN_mcs)
                                key_topN_data[idx_to_replace] = (mc,prefix)

                n_prefixes_analysed+=1

                end = default_timer()
                elapsed = end-start
                times.append(elapsed)

