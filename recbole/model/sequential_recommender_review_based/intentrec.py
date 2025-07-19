"""
INTENTREC : 
################################################

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder_with_crossattn
from recbole.model.loss import BPRLoss
import random


import torch.nn.functional as F


class IntentRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(IntentRec, self).__init__(config, dataset)

        # init
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']                     # same as embedding_size
        self.inner_size = config['inner_size']                       # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        
        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.ml  = config['ml']
        self.tau = config['tau']
        self.sim = config['sim']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        

        """
        * add_cross_attn added
        (recbole.model.layers) TransformerEncoder_with_crossattn > TransformerLayer_with_crossattn > MultiHeadAttention_with_crossattn
        """
        self.trm_encoder = TransformerEncoder_with_crossattn(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_cross_attn = True
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # review text related
        self.text_dropout = nn.Dropout(0.2)
        self.text_proj = nn.Linear(768, 64)
        
    
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # mask_default
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

        # nce_fct
        self.nce_fct = nn.CrossEntropyLoss()

        # ML # 64 > 2
        self.itm_head = nn.Linear(config['hidden_size'], 2)    

        # parameters initialization
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, # The above code is declaring a variable named "max_len" without
        # assigning it a value.
        max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask               #  [B, 1, L, L]  : visible keys differ for each query
    
    
    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask                #  [B, 1, 1, L]  : visible keys same for all queries
    
    # forward : + review_seq, cls_seq
    def forward(self, item_seq, item_seq_len, review_seq, cls_seq):
        
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # extended_attention_mask = self.get_attention_mask(item_seq)      
        extended_attention_mask = self.get_bi_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)         # """Gathers the vectors at the specific positions"""


        # Inference
        if review_seq is None :
            return output, None
            

        ################# get init cls_seq #################
        
        
        
        cls_seq = self.text_proj(cls_seq)       # 768 > 64

        cls_seq = cls_seq.reshape(-1, cls_seq.size()[-1])             # B * L, 64
        review_seq = review_seq.reshape(-1, review_seq.size()[-1])    # B * L, 256

        nan_matrix = torch.full_like(cls_seq, torch.nan)
        zero_matrix = torch.zeros_like(cls_seq)
        
        ind = review_seq.sum(dim=1) == 0
        
        # ml 사용
        cls_seq_aug = cls_seq.clone()        
        cls_seq_aug[ind] = zero_matrix[ind]   
        cls_seq_aug = cls_seq_aug.reshape(-1,item_seq.size()[1],64)      

        # filtering : cls_seq (no grad) 
        cls_seq[ind] = nan_matrix[ind]
        cls_seq = cls_seq.reshape(-1,item_seq.size()[1],64)     # reshape : B, L, 64
        
        # mean 
        intent_feat = cls_seq.nanmean(dim=1)                        # batch, 64


        #==================== (ML) matching loss ====================#

        B = item_seq.shape[0]

        if B < 2 :  # 예외처리
            
            return output, intent_feat, None
        

        
        #==================== HARD Sampling code  ===================#
        
        with torch.no_grad(): 
            
            weights_i_c, weights_c_i = self.get_sim(output, intent_feat, temp=self.tau, sim=self.sim)
        
        
        # Adopted the BLIP approach
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # select a negative cls seq (20, 64) for each item seq (20, 64)
        cls_neg = []    
        cls_atts_neg = []
        
        for b in range(B):
            neg_idx = torch.multinomial(weights_i_c[b], 1).item()
            cls_neg.append(cls_seq_aug[neg_idx])
            cls_atts_neg.append(extended_attention_mask[neg_idx])
        cls_neg      = torch.stack(cls_neg,dim=0)  
        cls_atts_neg = torch.stack(cls_atts_neg,dim=0)   

        # select a negative item seq for each cls
        item_neg = []
        item_atts_neg = []
        item_len_neg = []
        
        for b in range(B):
            neg_idx = torch.multinomial(weights_c_i[b], 1).item()
            item_neg.append(input_emb[neg_idx])
            item_atts_neg.append(extended_attention_mask[neg_idx])   # extended_attention_mask is identical
            item_len_neg.append(item_seq_len[neg_idx])               # identify preference position 
        item_neg = torch.stack(item_neg,dim=0)   
        item_atts_neg = torch.stack(item_atts_neg,dim=0)    
        item_len_neg  = torch.stack(item_len_neg,dim=0)  


        #############################################       #############################################

        # batch * 3 configuration
        item_all     = torch.cat([input_emb, input_emb, item_neg],dim=0) 
        item_att_all = torch.cat([extended_attention_mask, extended_attention_mask, item_atts_neg],dim=0) 
        # additional considerations
        item_seq_len_all = torch.cat([item_seq_len, item_seq_len, item_len_neg], dim=0) 
        
        cls_all      = torch.cat([cls_seq_aug, cls_neg, cls_seq_aug], dim=0) 
        cls_att_all  = torch.cat([extended_attention_mask, cls_atts_neg, extended_attention_mask], dim=0)

        
        multi_feat = self.trm_encoder(item_all, item_att_all, encoder_hidden_states=cls_all, encoder_attention_mask=cls_att_all, output_all_encoded_layers=True)
        
        multi_feat_target = multi_feat[-1]
        multi_feat_target = self.gather_indexes(multi_feat_target, item_seq_len_all - 1)   # B*3, 64
    
        
        return output, intent_feat, multi_feat_target  # [B H]        
        



    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]               # self.ITEM_SEQ : self.ITEM_ID + config['LIST_SUFFIX']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]       # self.ITEM_SEQ_LEN : item_length

        cls_seq = interaction['init_cls_list']
        review_seq = interaction['review_list']     

        seq_output, text_feat, cross_feat = self.forward(item_seq, item_seq_len, review_seq, cls_seq)   # forward pass here
        
        seq_output_rs, _ = self.forward(item_seq, item_seq_len, None, None)
        
        pos_items = interaction[self.POS_ITEM_ID]          
        
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output_rs * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_rs * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)

        # CE
        else:  
            
            test_item_emb = self.item_embedding.weight[:self.n_items]                    # unpad the augmentation mask
            logits = torch.matmul(seq_output_rs, test_item_emb.transpose(0, 1))          # transpose item embedding dims for dot product
            
            loss = self.loss_fct(logits, pos_items)                                      # self.loss_fct = nn.CrossEntropyLoss()
            
        
        ############################ + nce loss ############################
        
        #item_seq_output = seq_output.clone()
        nce_logits, nce_labels = self.info_nce(seq_output, text_feat, temp=self.tau, batch_size=seq_output.shape[0], sim=self.sim)

        nce_loss = self.nce_fct(nce_logits, nce_labels)
        
        
        ############################ + matching loss #######################

        # exception handling (considering batch size)
        if cross_feat == None :  
            return loss, self.lmd * nce_loss, 0
            

        cross_feat_output = self.itm_head(cross_feat)       # B, 64  > B, 2  

        cross_feat_labels = torch.cat([torch.ones(seq_output.shape[0], dtype=torch.long),torch.zeros(2 * seq_output.shape[0], dtype=torch.long)],
                                       dim=0).to(item_seq.device)   # create labels
        
        loss_itm = F.cross_entropy(cross_feat_output, cross_feat_labels)  

                 # 1: NI          # 2: CL              # 3: ML
        # return self.lmd * loss, self.lmd * nce_loss, self.ml * loss_itm        
        return loss, self.lmd * nce_loss, self.ml * loss_itm  
        
    
    # code modification
    def mask_correlated_samples(self, batch_size):
        
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        # mask = mask.fill_diagonal_(0)
        mask[:batch_size, :batch_size] = 0
        mask[batch_size:, batch_size:] = 0
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            
        return mask
    
    
    # NCE
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)   # logits
        return logits, labels
    
    
    # for neg & hard sampling
                  # item_seq_output(z_i), text_feat(z_j)
    def get_sim(self, z_i, z_j, temp, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
             
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos': 
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp                # B*2, 64 * 64, B*2 : B*2, B*2
    
        sim_i_j = sim[batch_size:, :batch_size]          # B,B
        sim_j_i = sim[:batch_size, batch_size:]
        
        weights_i_j = F.softmax(sim_i_j, dim=1) + 1e-4   # 1e-4 : refer bilp 
        weights_j_i = F.softmax(sim_j_i, dim=1) + 1e-4
        weights_i_j.fill_diagonal_(0) 
        weights_j_i.fill_diagonal_(0) 
        
        return weights_i_j, weights_j_i
    


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # review_seq = interaction['review_list'] review is not needed during inference
        seq_output, _ = self.forward(item_seq, item_seq_len, None, None)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    
    # eval
    def full_sort_predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # review_seq = interaction['review_list'] 추론시 리뷰 불필요
        
        seq_output, _ = self.forward(item_seq, item_seq_len, None, None)
        
        # self.n_items = dataset.num(self.ITEM_ID)
        test_items_emb = self.item_embedding.weight[:self.n_items]         # exclude mask token (self.n_items + 1, mask token)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]      
        return scores