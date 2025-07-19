"""
PIRSP
################################################

Reference:
    Zhang et al. "Improving current interest with item and review sequential patterns for sequential    
    recommendation."
    In 2021.

Reference code:
    The authors' pytorch implementation https://github.com/WHUIR/RNS
    * PIRSP is based on RNS with an added item sequence encoder block.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
# recbole : SequentialRecommender > type = ModelType.SEQUENTIAL 
from recbole.utils import InputType

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

from logging import getLogger



class PIRSP(SequentialRecommender):
    
    """
    __init__() input the parameters of config. and dataset, where config is used to input parameters, 
    dataset is leveraged to input datasets including n_users, n_items

    """
    
    input_type = InputType.PAIRWISE  # need for neg sampling
    
    
    def __init__(self, config, dataset):

        super(PIRSP, self).__init__(config, dataset)


        self.logger = getLogger()

        # load parameters info
        self.L = self.max_seq_length             # default 5, following RNS
        self.n_t = config["n_t"]                 # number of text cnn filters for each size (default=2)
        self.n_k = config["n_k"]                 # number of aspects (default=5)
        self.drop_ratio = config["drop_ratio"]   # 0.3
        self.ac_conv = activation_getter[config["ac_conv"]]         # relu
        self.ac_fc = activation_getter[config["ac_fc"]]             # relu

        self.w_vec_dim = config["vec_dim"]      # dimension of word embeddings (default=25)
        self.alpha = config["alpha"]            # weight of sequential preference (default=0.1)

        
        # load dataset info
        self.num_users = dataset.num(self.USER_ID)
        self.num_items = self.n_items


        self.loss_type = config["loss_type"]

        if self.loss_type == "BCE":
             self.loss_fct = BCELoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()

        
        # define layers and loss
        len_vocabulary = len(dataset.field2token_id['review'])
                               
        self.word_embed = nn.Embedding(len_vocabulary, self.w_vec_dim, padding_idx=0)
        self.word_embed.weight.data[1:].normal_(0, 1.0 / self.word_embed.embedding_dim)
        self.word_embed.weight.requires_grad = True


        # user_doc & item_doc
        user_doc = torch.LongTensor(np.stack([dataset.user_doc[i] for i in range(len(dataset.user_doc))]))
        item_doc = torch.LongTensor(np.stack([dataset.item_doc[i] for i in range(len(dataset.item_doc))]))

        self.user_doc = nn.Parameter(user_doc, requires_grad=False)
        self.item_doc = nn.Parameter(item_doc, requires_grad=False)

        # self.trans_mat = nn.ParameterList([nn.Parameter(torch.randn((self.w_vec_dim, self.w_vec_dim), dtype=torch.float), requires_grad=True) for _ in range(self.n_k)])        # 25,25 * 5 (list)
        self.trans_mat = nn.Parameter(torch.randn((5, self.w_vec_dim, self.w_vec_dim), dtype=torch.float), requires_grad=True)
        
        lengths = [1, 3, 5, 7, 9]
                                                    # 5       # 2                        
        self.conv_t_item = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim), bias=False) for i in lengths])
        self.conv_t_user = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim)) for i in lengths])
        
        self.k_dim = self.n_t * len(lengths)                  # 10
        self.full_dim = self.n_k * self.k_dim                 # 50
        self.pos_embed = nn.Embedding(self.L, self.k_dim)     # 4,10

        self.dropout = nn.Dropout(self.drop_ratio)

        # for PIRSP : emb dim 10 (must be same total filter size 10)
        self.item_embedding = nn.Embedding(
            self.n_items, 10, padding_idx=0         
        )

        # num_layers=1 (same GRU4Rec)
        self.gru_layers = nn.GRU(
            input_size=10,
            hidden_size=10,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        
        self.dense = nn.Linear(20, 10)

        """
        self.wi = nn.Linear(10, 10, bias=False)
        self.wi = nn.Linear(10, 10, bias=False)
        """
        self.a1 = torch.nn.Parameter(torch.randn(10,10), requires_grad=True)
        self.a2 = torch.nn.Parameter(torch.randn(10,10), requires_grad=True)

        self.wi = torch.nn.Parameter(torch.randn(10,10), requires_grad=True)
        self.wr = torch.nn.Parameter(torch.randn(10,10), requires_grad=True)
    
                      # L, W                         # L
    def forward(self, seq_var, user_var, item_repr, item_seq, item_seq_len):

        
        ############## item : gru ##############

        item_seq_emb = self.item_embedding(item_seq)    # Batch, L, 10
        
        if self.training :
            self.gru_layers.flatten_parameters()

        hj, _ = self.gru_layers(item_seq_emb)           # Batch, L, 10

        """ Attention and filtering """

        hL = self.gather_indexes(hj, item_seq_len - 1)
        
        # mm (B, 1, 10), (10, 10) > (B,1,10) + (B,L,10) > sig >  (B,L,10) * (B,L,10) : sum > (B,L)
        # self.logger.debug(hj.shape) 
        aLj = torch.sum(torch.sigmoid(torch.matmul(hL.unsqueeze(1), self.a1) + torch.matmul(hj, self.a2)) * item_seq_emb, dim=2)

        ahj = aLj.unsqueeze(-1) * hj
        
        # filtering
        zero_matrix = torch.zeros_like(ahj)
        inx = item_seq == 0
        # clone : inplace modification https://daeheepark.tistory.com/23
        #hj_clone   = hj.clone()        
        ahj_filter = ahj.clone()
        #hj_clone[inx]   = zero_matrix[inx]
        ahj_filter[inx] = zero_matrix[inx]

        hl = ahj_filter.sum(dim=1)            # (B,10)
        
        h = torch.cat([hl, hl], dim=1) # (B,20)

        p_i = self.dense(h)                   # (B,10) * FC in figure


        ################ RNS : review : CNN ################
        seq_var_word_vector = self.word_embed(seq_var)                     # (B, L, W, D)
        user_var_word_vector = self.word_embed(user_var.unsqueeze(1))      # (B, 1, W, D)  

        l1, l2 = [list() for _ in range(2)]

        seq_var_aspect_concat  = torch.matmul(seq_var_word_vector.unsqueeze(2), self.trans_mat)
        user_var_aspect_concat = torch.matmul(user_var_word_vector.unsqueeze(2), self.trans_mat)

        seq_var_aspect_concat = seq_var_aspect_concat.reshape(-1, seq_var_aspect_concat.shape[-3], seq_var_aspect_concat.shape[-2], seq_var_aspect_concat.shape[-1])
        
        for conv in self.conv_t_item:                    # list of model
                conv_out = self.ac_conv(conv(seq_var_aspect_concat).squeeze(3))  
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (B, 2, W'') > (B, 2, 1) > sq : (B, 2)
                
                l1.append(pool_out)
        seq_var_repr_origin= torch.cat(l1, 1)  
        seq_var_repr_origin = seq_var_repr_origin.reshape(seq_var.shape[0], seq_var.shape[1], -1)    # (B, L, 10)

        # get device
        device = seq_var.device 

        # Even if hj is not filtered: filtering is done later at the review stage
        # seq_var_repr += gru out + positional embedding : Batch, L, 10
        # qj = qj + poj + hj
        seq_var_repr = seq_var_repr_origin + hj + self.pos_embed(torch.arange(0, seq_var.shape[1]).to(device))
        # self.logger.debug(f"seq_var_repr {seq_var_repr.shape}")       # [128, 5, 10]
        

        ################ RNS : user (review : CNN ################   
        for conv in self.conv_t_user:

            conv_out = self.ac_conv(conv(user_var_aspect_concat.squeeze(1))).squeeze(3)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
            l2.append(pool_out)  
            
        user_repr = torch.cat(l2, 1)  
        

        """
        (note)
        # print(seq_var_repr.shape, item_repr.shape, user_repr.shape) 
        # torch.Size([256, 5, 10]) torch.Size([256, item, 10]) torch.Size([256, 10])
        """        

        ######################################## phrase 2 ########################################  
        
        # item_seq attn
        ind = seq_var_repr_origin.sum(dim=2) == 0         # find zeros in last dimension, before positional embedding

        # self.logger.debug(ind.shape)    # [B, L]
        
        temp = torch.sum(seq_var_repr.unsqueeze(1) * item_repr.unsqueeze(2), 3)    # [B, 1, L, 10] * [B, item, 1, 10] > sum [B, item, L] 
        # self.logger.debug(temp.shape)     # [B, item, L])
        ind = ind.unsqueeze(1).expand(-1,temp.shape[1],-1)      # [B,item,L]
        temp[ind] += -1e5

        w = F.softmax(temp, dim=2)  

        # u
        u = torch.sum(seq_var_repr.unsqueeze(1) * w.unsqueeze(-1), 2)      # [B,1,L,10] * [B,item,L,1] > sum [B, item, 10]
        # self.logger.debug(u.shape)     # [B,item,L]

        # m
        m = torch.argmax(w, 2).unsqueeze(2)              # torch.Size([256, item, 1])
        index = m.expand(-1, -1, seq_var_repr.shape[2])  # torch.Size([256, item, 10]): index of the most important item
        
        p = seq_var_repr.gather(1, index)                # gather operation: expand 10 > means to get all
        # self.logger.debug(p.shape)                     # [B, item, 10]
        
        w2_u = torch.sum(u * item_repr, 2).unsqueeze(2)  # [B, item, 10]: preference per item * [B, item, 10]: items per batch > [B, item, 1]

        w2_m = torch.sum(m * item_repr, 2).unsqueeze(2)   

        w2 = F.softmax(torch.cat((w2_u, w2_m), dim=2),2) # [B, item, 2]
        

        p_r = (w2[:,:,0].unsqueeze(2) * u) + (w2[:,:,1].unsqueeze(2) * m)  # [B, item, 10]
            

        """
        p_item torch.Size([B, 10])
        p_review torch.Size([B, ITEM, 10])
        """

        # item - review gating
        wpi = torch.matmul(p_i, self.wi) # 64, 10       > 64, 10     
        wpr = torch.matmul(p_r, self.wr) # 64, item, 10 > 64, item, 10 

        f = torch.sigmoid(wpi.unsqueeze(1) + wpr)          # 64, item, 10 

        
        seq_repr = (f * p_i.unsqueeze(1)) + ((1-f) * p_r)  # 64, item, 10 
        #################################################
        
        res = torch.sum((self.alpha * seq_repr + user_repr.unsqueeze(1)) * item_repr, 2)
    
        
        return res   # items_prediction_ per item


    def calculate_loss(self, interaction):
        
        #seq_var  = interaction['item_id_aggreview_list']
        #user_var = interaction['user_id_aggreview']
        #item_var = interaction['item_id_aggreview']
        #device = interaction['item_id_list'].device

        item_seq = interaction['item_id_list']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        pos_items = interaction[self.POS_ITEM_ID] # 'item_id'

        # torch.cat([torch.tensor([1]) , torch.tensor([1])])
        # ind = torch.cat([pos_items, torch.LongTensor[0]])
        
        # torch.LongTensor(model.item_doc).unsqueeze(0).expand(256,-1,-1)
        #item_var = torch.LongTensor(self.item_doc).unsqueeze(0).expand(seq_var.shape[0],-1,-1).to(device)


        if self.loss_type == "BCE":

            neg_items = interaction[self.NEG_ITEM_ID] # 'neg_item_id'
            
            pos_item_var  = self.item_doc[pos_items].unsqueeze(1)               # B,1,L
            neg_item_var  = self.item_doc[neg_items]                            # B,4,L

            item_ids = torch.cat((pos_items.unsqueeze(1), neg_items), 1)
            items_to_predict = torch.cat((pos_item_var, neg_item_var), 1)

            """
            item_seq.shape
            torch.Size([2, 5])
            items_to_predict.shape
            torch.Size([2, 5, 100])
            """

            item_repr = self._get_item_repr(items_to_predict, item_ids)
            seq_output = self.forward(seq_var, user_var, item_repr, item_seq, item_seq_len)          # B,5

            targets_prediction, negatives_prediction =\
            torch.split(seq_output,[pos_item_var.size(1),neg_item_var.size(1)], dim=1)

            loss = self.loss_fct(targets_prediction, negatives_prediction)
            
            return loss
        
            
        else:
               
            device = item_seq.device

            item_var = self.item_doc.unsqueeze(0)
            item_ids = torch.arange(0, self.num_items).unsqueeze(0).to(device)   
            item_repr = self._get_item_repr(item_var, item_ids)
            item_repr = item_repr.expand(seq_var.shape[0],-1,-1)  # expand batch
            
            scores = self.forward(seq_var, user_var, item_repr, item_seq, item_seq_len)   # [B n_items]
            loss = self.loss_fct(scores, pos_items)
            return loss
            
            # loss = self.loss_fct(seq_output, pos_items)
            # return loss
        
    # Improve training and inference speed
    def _get_item_repr(self, item_var, item_id):

        item_var_word_vector = self.word_embed(item_var)                   # (B, C, W, D)  # candidate items

        # (1, C, W, D) > (1, C, 1, W, D) * (5, D, D') > (1, C, 5, W, D')

        item_var_aspect_concat = torch.matmul(item_var_word_vector.unsqueeze(2), self.trans_mat)
        item_var_aspect_concat = item_var_aspect_concat.reshape(-1, item_var_aspect_concat.shape[-3], item_var_aspect_concat.shape[-2], item_var_aspect_concat.shape[-1])

        l = []

        
        for conv in self.conv_t_item:                    # list of model
                conv_out = self.ac_conv(conv(item_var_aspect_concat).squeeze(3))  
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)       # (1, 2, W'') > (1, 2, 1) > sq : (1, 2)
                l.append(pool_out)
            
        item_repr = torch.cat(l, 1)         
        item_repr = item_repr.reshape(item_var.shape[0], item_var.shape[1], -1)

        # add item_embedding
        item_repr += self.item_embedding(item_id)  
        
        return item_repr         # (B, C, 10)      >>>>>>>> Item candidates

    
    def predict(self, interaction):

        item_seq = interaction['item_id_list']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        # Note: item_var / item_ids

        device = item_seq.device
        
        item_var = self.item_doc.unsqueeze(0)                                # 1,C,100
        item_ids = torch.arange(0, self.num_items).unsqueeze(0).to(device)   # 1,C
        
        item_repr = self._get_item_repr(item_var, item_ids)
        item_repr = item_repr.expand(seq_var.shape[0],-1,-1)  # expand batch       

        scores = self.forward(seq_var, user_var, item_repr, item_seq, item_seq_len)      # B,All items

        return scores

    # get score
    def full_sort_predict(self, interaction):


        item_seq = interaction['item_id_list']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        # Note: item_var / item_ids

        device = item_seq.device
        
        item_var = self.item_doc.unsqueeze(0)                                # 1,C,100
        item_ids = torch.arange(0, self.num_items).unsqueeze(0).to(device)   # 1,C
        
        item_repr = self._get_item_repr(item_var, item_ids)
        item_repr = item_repr.expand(seq_var.shape[0],-1,-1)  # expand batch       

        scores = self.forward(seq_var, user_var, item_repr, item_seq, item_seq_len)      # B,All items

        return scores