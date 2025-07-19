"""
CCA
################################################

Reference:
    Bingsen Huang et al. "Cascaded Cross Attention for Review-based Sequential Recommendation."
    In ICDM 2023.

Reference code:
    The authors' pytorch implementation https://github.com/BING303/CCA
"""


import torch
from torch import nn
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CCALayer
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

class CCA(SequentialRecommender):

    input_type = InputType.PAIRWISE  # required for negative sampling

    def __init__(self, config, dataset):

        
        super(CCA, self).__init__(config, dataset)

        self.dataset = dataset
        
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        #self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.cca_encoder = CCALayer(
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )            
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BCE":
            self.loss_fct = BCELoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BCE', 'CE']!")

        # https://github.com/BING303/CCA/blob/main/CCA.py#L794
        self.emb_1 = nn.Linear(774, self.hidden_size * 5)  # 768 + 6
        # https://github.com/BING303/CCA/blob/main/CCA.py#L811
        self.emb_2 = nn.Linear(self.hidden_size * 6, self.hidden_size)
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, review_cls_seq, item_cls_seq, time_seq, pos_item_emb, neg_item_emb) : 


        item_seq_emb = self.item_embedding(item_seq)  # item embeddings
    
        """
        # https://github.com/BING303/CCA/blob/main/CCA.py#L784
        # gate
        W_s = tf.Variable(tf.random.normal(shape=(embedding_size, embedding_size),stddev=0.01,mean=0,dtype=tf.float32)) 
        W_m = tf.Variable(tf.random.normal(shape=(embedding_size, embedding_size),stddev=0.01,mean=0,dtype=tf.float32))
        b_g = tf.Variable(tf.random.normal(shape=(embedding_size),stddev=0.01,mean=0,dtype=tf.float32))
        one = tf.constant(1.0)

        g = tf.sigmoid(tf.matmul(self.seq_feat_single,W_s) + tf.matmul(self.seq_feat_mean,W_m) + b_g)
        self.seq_feat = tf.multiply(g,self.seq_feat_single) + tf.multiply((one-g),self.seq_feat_mean)
        """
        device = item_seq.device
        W_s = nn.Parameter(torch.randn(768, 768) * 0.01).to(device)  # requires_grad=True
        W_m = nn.Parameter(torch.randn(768, 768) * 0.01).to(device)
        b_g = nn.Parameter(torch.randn(768) * 0.01).to(device)


        g = torch.sigmoid(torch.matmul(review_cls_seq, W_s) + torch.matmul(item_cls_seq, W_m) + b_g)
        review_feat = g * review_cls_seq + (1 - g) * item_cls_seq   # A

        
        """
        self.seq_feat_in = tf.concat([self.seq_feat , self.seq_cxt], -1)
        self.seq_feat_emb = tf.layers.dense(inputs=self.seq_feat_in, units=args.hidden_units*5,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        """
        # time_seq : # C                         
        seq_feat_in = torch.cat((review_feat, time_seq), dim=-1)     
        seq_feat_emb = self.emb_1(seq_feat_in)                      # A & C (cat) > Z

        """
        self.seq_concat = tf.concat([self.seq_in , self.seq_feat_emb], 2)
        self.seq = tf.layers.dense(inputs=self.seq_concat, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp')
        """
        seq_concat = torch.cat((item_seq_emb, seq_feat_emb), dim=-1)  
        seq_emb = self.emb_2(seq_concat)                           # A & Z (cat) > hat E

        # encoder input
        input_emb = self.LayerNorm(seq_emb)  # https://github.com/BING303/CCA/blob/main/CCA.py#L830
        input_emb = self.dropout(input_emb)  # https://github.com/BING303/CCA/blob/main/CCA.py#L820

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output_all = trm_output[-1]
        # output_1
        output_1 = self.gather_indexes(output_all, item_seq_len - 1)   # B,64

        
        # https://github.com/BING303/CCA/blob/main/CCA.py#L966 : Only 1 layer 
        
        if pos_item_emb is None :
            return output_1, output_all

        # for neg sampling
        pos_item_emb = pos_item_emb.unsqueeze(1)
        neg_item_emb = neg_item_emb.unsqueeze(1)
                                                      
                                        # (Q) item_emb, (K, V) output of 1st encoder
        output_2_pos = self.cca_encoder(pos_item_emb, None, output_all, extended_attention_mask).squeeze()   #[B]
        output_2_neg = self.cca_encoder(neg_item_emb, None, output_all, extended_attention_mask).squeeze()   #[B]

        # final output [reference: https://github.com/BING303/CCA/blob/main/CCA.py#L996 > just sum]
        
        return output_1, output_all, output_2_pos, output_2_neg


    """
    CE loss implementation requires high GPU computation
    """
    def calculate_loss(self, interaction):
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        review_cls_seq = interaction["init_cls_list"]
        item_cls_seq = interaction["item_cls_mean_list"]
        time_seq = interaction["time_emb_list"]
        
        pos_items = interaction[self.POS_ITEM_ID]
        time_emb  = interaction["time_emb"]       # time_emb of pos_items 
        
        
        if self.loss_type == "BCE" :  # neg sampling
            
            neg_items = interaction[self.NEG_ITEM_ID].squeeze(1)      # [B]
            pos_item_emb, neg_item_emb = self._get_item_repr(time_emb, pos_items, neg_items)    # get emb after concat & linear : [B, 64]
            
            seq_output, _, output_2_pos, output_2_neg = self.forward(item_seq, item_seq_len, review_cls_seq, item_cls_seq, time_seq, pos_item_emb, neg_item_emb)
            
            pos_score = torch.sum(seq_output * pos_item_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_item_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            loss2 = self.loss_fct(output_2_pos, output_2_neg)
            
            return loss, loss2
            
        else:  # CE has issues from computational perspective
        
            # test_item_emb = self.item_embedding.weight
            test_item_emb = self._get_all_item_repr(time_emb, None)  # [B, all_item, 64]

            seq_output, output_all = self.forward(item_seq, item_seq_len, review_cls_seq, item_cls_seq, time_seq, None, None)

            extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
            output_2 = self.cca_encoder(test_item_emb, None, output_all, extended_attention_mask) # [B,all_item,1]

            seq_output = seq_output.unsqueeze(1)  # [B,1,64]   
            # batched MM      
            logits = torch.matmul(seq_output, test_item_emb.transpose(1, 2)).squeeze(1)  # [B,1,item] > [B,item]
            loss = self.loss_fct(logits, pos_items)
            
            logits_2 = output_2.squeeze(2)              # [B,all_item]
            loss2 = self.loss_fct(logits_2, pos_items)

            
            return loss

                
    def _get_item_repr(self, time_emb, pos_items, neg_items) :  # received time_emb 

        device = time_emb.device  # get device
        item_cls_emb = torch.Tensor(np.array(list(self.dataset.item_cls_dict.values()))).to(device)  # item_cls_dict {item id:item_cls_mean_list}: [item, 768]
  
        pos_items_emb = self.item_embedding(pos_items)  # E
        neg_items_emb = self.item_embedding(neg_items)  # E 

        # use only agg reviews
        feat_in_pos  = torch.cat((item_cls_emb[pos_items], time_emb), dim=-1)   # review_feat[indexing] > [B, 768], [B, 6]
        feat_emb_pos = self.emb_1(feat_in_pos)                                  # [B, 320]
        concat_pos   = torch.cat((pos_items_emb, feat_emb_pos), dim=-1)         # [B, 384]
        emb_pos      = self.emb_2(concat_pos)                                   # hat E  > [B, 64]

        # with the same timestamp as their corresponding positive test item : https://github.com/BING303/CCA/blob/main/CCA.py#L906
        feat_in_neg  = torch.cat((item_cls_emb[neg_items], time_emb), dim=-1)   
        feat_emb_neg = self.emb_1(feat_in_neg)                             
        concat_neg   = torch.cat((neg_items_emb, feat_emb_neg), dim=-1)          
        emb_neg      = self.emb_2(concat_neg)                                                      

        return emb_pos, emb_neg       # [B, 64] > each item


    def _get_all_item_repr(self, time_emb, target_items) :   # received time_emb 

        device = time_emb.device
        item_cls_emb = torch.Tensor(np.array(list(self.dataset.item_cls_dict.values()))).to(device)  # [item, 768]

        if target_items :
            items_emb = self.item_embedding.weight[target_items]
            item_cls_emb = item_cls_emb[target_items]
        else :
            items_emb = self.item_embedding.weight                  # all items

        
        time_emb = time_emb.unsqueeze(1)
        feat_in_pos  = torch.cat((item_cls_emb.expand(time_emb.shape[0],-1,-1), time_emb.expand(-1,item_cls_emb.shape[0],-1)), dim=-1)   # cat([B, item, 768], [B, item, 6]) : 배치/아이템만큼 확장
        feat_emb_pos = self.emb_1(feat_in_pos)                                                                                           # [B, item, 64*5]      
        concat_pos   = torch.cat((items_emb.expand(time_emb.shape[0],-1,-1), feat_emb_pos), dim=-1)                                      # [B, item, 64], [B, item, 64*5]       
        emb_pos      = self.emb_2(concat_pos)                                                                                            # [B, item, 64]

        return emb_pos            # B, item, 64
 
    
    def predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        review_cls_seq = interaction["init_cls_list"]
        item_cls_seq = interaction["item_cls_mean_list"]
        time_seq = interaction["time_emb_list"]
        
        time_emb  = interaction["time_emb"]
        test_item = interaction[self.ITEM_ID]


        seq_output, output_all = self.forward(item_seq, item_seq_len, review_cls_seq, item_cls_seq, time_seq, None, None)
        seq_output = seq_output.unsqueeze(1)
        test_item_emb = self._get_all_item_repr(time_emb, test_item)
        
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        output_2 = self.cca_encoder(test_item_emb, None, output_all, extended_attention_mask) # B,item,1
        
        scores = torch.mul(seq_output, test_item_emb).sum(dim=-1)  # [B, n_items]

        scores_1 = torch.matmul(seq_output, test_item_emb.transpose(1, 2)).squeeze(1)  # [B,1, n_items] : batched MM > squeeze
        scores_2 = output_2.squeeze(2)    
    
        scores = scores_1 + scores_2
    
        return scores

    def full_sort_predict(self, interaction):
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        review_cls_seq = interaction["init_cls_list"]
        item_cls_seq = interaction["item_cls_mean_list"]
        time_seq = interaction["time_emb_list"]
        
        time_emb  = interaction["time_emb"]

        seq_output, output_all = self.forward(item_seq, item_seq_len, review_cls_seq, item_cls_seq, time_seq, None, None)
        seq_output = seq_output.unsqueeze(1)     # [B,1,64]
        test_item_emb = self._get_all_item_repr(time_emb, None)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        output_2 = self.cca_encoder(test_item_emb, None, output_all, extended_attention_mask) # B,item,1
        
        
        scores_1 = torch.matmul(seq_output, test_item_emb.transpose(1, 2)).squeeze(1)  # [B,1, n_items] : batched MM > squeeze
        scores_2 = output_2.squeeze(2)

        scores = scores_1 + scores_2
        return scores