"""
RNS
################################################

Reference:
    Chenliang Li et al. "A Review-Driven Neural Model for Sequential Recommendation."
    In IJCAI 2019.

Reference code:
    The authors' pytorch implementation https://github.com/WHUIR/RNS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType
from logging import getLogger


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

class RNS(SequentialRecommender):
    
    """
    __init__() input the parameters of config. and dataset, where config is used to input parameters, 
    dataset is leveraged to input datasets including n_users, n_items

    """
    
    input_type = InputType.PAIRWISE  # required for negative sampling
    
    
    def __init__(self, config, dataset):

        super(RNS, self).__init__(config, dataset)

        self.logger = getLogger()
        """
        reference code
        #######################################
        self.args = model_args
        L = self.args.L
        self.n_t = self.args.nt   # number of text cnn filters for each size (default=2)
        self.n_k = self.args.nk   # number of aspects (default=5)
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        self.num_users = num_users
        self.num_items = num_items
        self.word2vec = None
        self.w_vec_dim = self.args.dim
        self.alpha = self.args.alpha
        

        reference arg
        #######################################
        # config
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_root', type=str, default='data/reviews_Amazon_Instant_Video.json/video_train.csv')
        parser.add_argument('--test_root', type=str, default='data/reviews_Amazon_Instant_Video.json/video_test.csv')
        parser.add_argument('--L', type=int, default=5)
        parser.add_argument('--T', type=int, default=1)
        parser.add_argument('--n_iter', type=int, default=5)
        parser.add_argument('--seed', type=int, default=2018)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--l2', type=float, default=1e-4)
        parser.add_argument('--neg_samples', type=int, default=3)
        parser.add_argument('--test_neg', type=int, default=100)
        parser.add_argument('--use_cuda', type=str2bool, default=True)
        config = parser.parse_args(args=[])
        
        # model_config
        model_parser = argparse.ArgumentParser()
        model_parser.add_argument('--drop', type=float, default=0.3)
        model_parser.add_argument('--ac_conv', type=str, default='relu')
        model_parser.add_argument('--ac_fc', type=str, default='relu')
        model_parser.add_argument('--dim', type=int, default=25, help='dimension of word embeddings')
        model_parser.add_argument('--nt', type=int, default=2, help='number of text cnn filters for each size')
        model_parser.add_argument('--nk', type=int, default=5, help='number of aspects')
        model_parser.add_argument('--alpha', type=float, default=0.1, help='weight of sequential preference')
        model_config = model_parser.parse_args(args=[])
        
        model_config.L = config.L  # add --L from config
        set_seed(config.seed, cuda=config.use_cuda
        """

        # load parameters info
        self.L = self.max_seq_length             
        self.n_t = config["n_t"]                 # number of text CNN filters for each size (default=2)
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
        """
        self.word_embed = nn.Embedding(len(vocabulary), self.w_vec_dim)
        self.word_embed.weight.data.normal_(0, 1.0 / self.word_embed.embedding_dim)
        self.word_embed.weight.requires_grad = True
        """

        len_vocabulary = len(dataset.field2token_id['review'])
        self.logger.debug(f"Model - len_vocabulary: {len(dataset.field2token_id['review'])}")
        self.logger.debug(f"   dataset.field2token_id['review']['[PAD]'] : (index) {dataset.field2token_id['review']['[PAD]']}")
                               
        self.word_embed = nn.Embedding(len_vocabulary, self.w_vec_dim, padding_idx=0)
        self.word_embed.weight.data[1:].normal_(0, 1.0 / self.word_embed.embedding_dim)
        self.word_embed.weight.requires_grad = True


        
        # user_doc & item_doc

        user_doc = torch.LongTensor(np.stack([dataset.user_doc[i] for i in range(len(dataset.user_doc))]))
        item_doc = torch.LongTensor(np.stack([dataset.item_doc[i] for i in range(len(dataset.item_doc))]))

        self.user_doc = nn.Parameter(user_doc, requires_grad=False)
        self.item_doc = nn.Parameter(item_doc, requires_grad=False)

        # self.trans_mat = nn.ParameterList([nn.Parameter(torch.randn((self.w_vec_dim, self.w_vec_dim), dtype=torch.float), requires_grad=True) for _ in range(self.n_k)])        # 25,25 * 5 (list)
        self.trans_mat = nn.Parameter(torch.randn((5, self.w_vec_dim, self.w_vec_dim), dtype=torch.float), requires_grad=True)   # [5,25,25]
        
        lengths = [1, 3, 5, 7, 9]
                                                    # 5       # 2                        
        self.conv_t_item = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim), bias=False) for i in lengths])
        self.conv_t_user = nn.ModuleList([nn.Conv2d(self.n_k, self.n_t, (i, self.w_vec_dim)) for i in lengths])
        
        self.k_dim = self.n_t * len(lengths)                  # 10
        self.full_dim = self.n_k * self.k_dim                 # 50
        self.pos_embed = nn.Embedding(self.L, self.k_dim)     # 4,10

        self.dropout = nn.Dropout(self.drop_ratio)


    
    """
    forward
    --------------------------------------------------------------------------
    (arg) 
    seq_var, user_var, item_var(※ items_to_predict)
    """

    def forward(self, seq_var, user_var, item_repr):


        seq_var_word_vector = self.word_embed(seq_var)                     # (B, L, W, D)  
        user_var_word_vector = self.word_embed(user_var.unsqueeze(1))      # (B, 1, W, D)  

        l1, l2 = [list() for _ in range(2)]
        
        #seq_var_aspect_concat = None
        #user_var_aspect_concat = None

        """Efficiency improvement
           seq :  (B, L, W, D) > (B, L, 1, W, D) * (5, D, D') > (B, L, 5, W, D')
           user : (B, 1, W, D) > (B, 1, 1, W, D) * (5, D, D') > (B, 1, 5, W, D') 
        
        [Original code]   
        for i in range(self.n_k):  # perform 5 times (aspect: 5)
            
            # matmul
                                                 #  (B, L, W, D) * (D, D') > unsqueeze(1) : (B, 1, L, W, D')
            seq_var_word_trans_vector = torch.einsum("abcd,de->abce", (seq_var_word_vector, self.trans_mat[i]))\
                .unsqueeze(1)
            # concatenate with unsqueezed dimension         #  (B, 5, L, W, D')
            seq_var_aspect_concat = torch.cat((seq_var_aspect_concat, seq_var_word_trans_vector), 1) \
                if seq_var_aspect_concat is not None else seq_var_word_trans_vector

                                                 #  (B, 1, W, D) * (D, D') > unsqueeze(1) : (B, 1, 1, W, D')
            user_var_word_trans_vector = torch.einsum("abcd,de->abce", (user_var_word_vector, self.trans_mat[i])) \
                .unsqueeze(1)
                                                 #  (B, 5, 1, W, D')
            user_var_aspect_concat = torch.cat((user_var_aspect_concat, user_var_word_trans_vector), 1) \
                if user_var_aspect_concat is not None else user_var_word_trans_vector       
        """
        seq_var_aspect_concat  = torch.matmul(seq_var_word_vector.unsqueeze(2), self.trans_mat)
        user_var_aspect_concat = torch.matmul(user_var_word_vector.unsqueeze(2), self.trans_mat)

        # Consider L
        """Efficiency improvement
        * equivalent to having 5 CNNs total
        (B, L, 5, W, D') > reshape (B*L, 5, W, D') > conv (B*L, 2, W, 1) > sq (B*L, 2, W') > mp (B*L, 2, 1) > sq (B*L, 2) > cat (B*L, 10)
        > reshape (B, L, 10)
        
        
        for j in range(seq_var_aspect_concat.shape[2]):
            s = seq_var_aspect_concat[:, :, j, :, :]         # selected dimension (reduced)
    
            for conv in self.conv_t_item:                    # list of model
                conv_out = self.ac_conv(conv(s).squeeze(3))  # (B, 5, W, D') > conv > (B, 2, W'',1) > sq (B, 2, W'')
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (B, 2, W'') > (B, 2, 1) > sq : (B, 2)
 
                l1.append(pool_out)
            s1 = torch.cat(l1, 1).unsqueeze(1)                # (B, 1, 10)      

            l2.append(s1)                      
            l1 = []                                            # init

            
        seq_var_repr_origin = torch.cat(l2, 1)         # (B, L, 10)          >>>>>>> short-term P        
        """
        seq_var_aspect_concat = seq_var_aspect_concat.reshape(-1, seq_var_aspect_concat.shape[-3], seq_var_aspect_concat.shape[-2], seq_var_aspect_concat.shape[-1])
        
        for conv in self.conv_t_item:                    # list of model
                conv_out = self.ac_conv(conv(seq_var_aspect_concat).squeeze(3))  
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (B, 2, W'') > (B, 2, 1) > sq : (B, 2)
                
                l1.append(pool_out)
        seq_var_repr_origin= torch.cat(l1, 1)  
        seq_var_repr_origin = seq_var_repr_origin.reshape(seq_var.shape[0], seq_var.shape[1], -1) 

        """
        (note)
        s torch.Size([64, 5, 100, 25])
        conv_out torch.Size([64, 2, 100])
        pool_out torch.Size([64, 2])
        conv_out torch.Size([64, 2, 98])
        pool_out torch.Size([64, 2])
        conv_out torch.Size([64, 2, 96])
        pool_out torch.Size([64, 2])
        conv_out torch.Size([64, 2, 94])
        pool_out torch.Size([64, 2])
        conv_out torch.Size([64, 2, 92])
        pool_out torch.Size([64, 2])
        s1 torch.Size([64, 1, 10])
        """

        # get device
        device = seq_var.device 

        # positional embedding
        seq_var_repr = seq_var_repr_origin + self.pos_embed(torch.arange(0, seq_var.shape[1]).to(device))

        # phrase 2      
        """
        Efficiency improvement : for loop is necessary
        (B, 1, 5, W, D') > sq (B, 5, W, D') > conv  (B, 2, W', 1) > sq (B, 2, W') > mp (B,2,1) > (B,2) > cat (B,10)
        
        l4 = []
        for conv in self.conv_t_user:

                        # (B, 5, 1, W, D') > # (B, 2, W', 1)
            conv_out = self.ac_conv(conv(user_var_aspect_concat.squeeze(2))).squeeze(3)
 

                       # (B, 2, W') > (B, 1, 1) > (B,1)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
            l4.append(pool_out)  
        user_repr = torch.cat(l4, 1)          
        """
        
        for conv in self.conv_t_user:

            conv_out = self.ac_conv(conv(user_var_aspect_concat.squeeze(1))).squeeze(3)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) 
            l2.append(pool_out)  
            
        user_repr = torch.cat(l2, 1)                 
         


        """
        (note)
        # print(seq_var_repr.shape, item_repr.shape, user_repr.shape) 
        # torch.Size([256, 4, 10]) torch.Size([256, 10, 10]) torch.Size([256, 10])
        """        

        '''  
        # attn
        al = []
                 
        # for each item: consider candidates
        for i in range(item_repr.shape[1]):
            s = item_repr[:, i, :].unsqueeze(1)               # each item [256,10] > [256,1,10]

  
            # filter to handle pad items
            ind = seq_var_repr_origin.sum(dim=2) == 0         # find zeros in last dimension, before positional embedding
            
            """
            temp = torch.sum(seq_var_repr * s, 2)
            temp[ind] += -1e5
            
            w = F.softmax(temp, 1).unsqueeze(2)   
            """
            temp = torch.sum(seq_var_repr * s, 2)
            temp[ind] += -1e5
            
            w = F.softmax(temp, 1).unsqueeze(2)   
            # u
            u = torch.sum(seq_var_repr * w, 1).unsqueeze(1) 
            # print(u.shape) torch.Size([256, 1, 10])
            # m
            m = torch.argmax(w, 1).unsqueeze(1)  
            index = m.expand(-1, 1, seq_var_repr.size(2))  # repetition
            # print(m.shape) torch.Size([256, 1, 1])
            # print(index.shape) torch.Size([256, 1, 10]

            
            p = seq_var_repr.gather(1, index) 
            p_u = torch.cat((u, p), 1)
            w2 = F.softmax(torch.sum(p_u * s, 2), 1).unsqueeze(2)      # 128, 2 , 1
            ss = torch.sum(p_u * w2, 1).unsqueeze(1)
            al.append(ss)

            
        seq_repr = torch.cat(al, 1) 
        '''
        
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
        index = m.expand(-1, -1, seq_var_repr.shape[2])  # torch.Size([256, item, 10]) : most important item index
        
        p = seq_var_repr.gather(1, index)                # gather operation: expand 10 > get all
        # self.logger.debug(p.shape)                     # [B, item, 10]
        
        w2_u = torch.sum(u * item_repr, 2).unsqueeze(2)  # [B, item, 10] : item preference * [batch_size, item, 10]: items per batch > [batch_size, item, 1]

        w2_m = torch.sum(m * item_repr, 2).unsqueeze(2)   

        w2 = F.softmax(torch.cat((w2_u, w2_m), dim=2),2) # [B, item, 2]
        

        seq_repr = (w2[:,:,0].unsqueeze(2) * u) + (w2[:,:,1].unsqueeze(2) * m)  # [B, item, 10] 

        
        """
        if for_pred:
            res = torch.sum((self.alpha * seq_repr + user_repr.unsqueeze(1)).squeeze() * item_repr.squeeze(), 1)
        else:                # add probability values > return probability for each item
        """
        # [B, item, 10] + [B, 1, 10] > [B, item, 10] * [B, item, 10] > [B, item, 1]
        res = torch.sum((self.alpha * seq_repr + user_repr.unsqueeze(1)) * item_repr, 2)  
    
        
        return res   # item predictions per item


    # follow for recbole

    def calculate_loss(self, interaction):
        
        #seq_var  = interaction['item_id_aggreview_list']
        #user_var = interaction['user_id_aggreview']
        #item_var = interaction['item_id_aggreview']

        #device = interaction['item_id_list'].device
    
        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        
        # print(seq_output)       # (Batch, Item_prob) : before sigmoid
        
        pos_items = interaction[self.POS_ITEM_ID] # 'item_id'




        # torch.cat([torch.tensor([1]) , torch.tensor([1])])

        # ind = torch.cat([pos_items, torch.LongTensor[0]])
        



        # torch.LongTensor(model.item_doc).unsqueeze(0).expand(256,-1,-1)
        #item_var = torch.LongTensor(self.item_doc).unsqueeze(0).expand(seq_var.shape[0],-1,-1).to(device)


        if self.loss_type == "BCE":
            neg_items = interaction[self.NEG_ITEM_ID] # 'neg_item_id'
            pos_item_var  = self.item_doc[pos_items].unsqueeze(1)               # B,1,L
            neg_item_var  = self.item_doc[neg_items]                            # B,4,L

            items_to_predict = torch.cat((pos_item_var, neg_item_var), 1)

            item_repr = self._get_item_repr(items_to_predict)
            seq_output = self.forward(seq_var, user_var, item_repr)             # B,5

            targets_prediction, negatives_prediction =\
            torch.split(seq_output,[pos_item_var.size(1),neg_item_var.size(1)], dim=1)

            #print("prediction.shape")
            #print(targets_prediction.shape)
            #print(negatives_prediction.shape)

            loss = self.loss_fct(targets_prediction, negatives_prediction)
            
            return loss
        

        elif self.loss_type == "BPR":
            """ must pos/neg 1:1"""

            pos_item_var  = self.item_doc[pos_items].unsqueeze(1)       
            neg_item_var  = self.item_doc[neg_items].unsqueeze(1) 

            pos_item_repr = self._get_item_repr(pos_item_var)
            neg_item_repr = self._get_item_repr(neg_item_var)
    
            pos_output = self.forward(seq_var, user_var, pos_item_repr)
            neg_output = self.forward(seq_var, user_var, neg_item_repr)
            

            loss = self.loss_fct(pos_output, neg_output)
            return loss
            
        else:  # self.loss_type = 'CE'
               # test_item_emb = self.item_embedding.weight
               # logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            item_var = self.item_doc.unsqueeze(0)
            item_repr = self._get_item_repr(item_var)
            item_repr = item_repr.expand(seq_var.shape[0],-1,-1)  # expand batch 
            
            scores = self.forward(seq_var, user_var, item_repr)   # [B n_items]
            loss = self.loss_fct(scores, pos_items)
            return loss
        



    # speed up train & infer
    def _get_item_repr(self, item_var):


        item_var_word_vector = self.word_embed(item_var)                   # (1, C, W, D)  # candidates

        '''
        ll1, ll2, ll3 = [list() for _ in range(3)]
        
        item_var_aspect_concat = None

        # this can be replaced
        for i in range(self.n_k):  # perform 5 times (aspect: 5)
            
            # matmul
            item_var_word_trans_vector = torch.einsum("abcd,de->abce", (item_var_word_vector, self.trans_mat[i])) \
                .unsqueeze(1)
                                                 #  (B, 5, C, W, D')
            item_var_aspect_concat = torch.cat((item_var_aspect_concat, item_var_word_trans_vector), 1) \
                if item_var_aspect_concat is not None else item_var_word_trans_vector

        
        # consider C                            # (B, 5, C, W, D')  
        for k in range(item_var.shape[1]):      
            ss = item_var_aspect_concat[:, :, k, :, :]
            for conv in self.conv_t_item:
                # (B, 5, 1, W, D') > # (B, 5, W, D')
                conv_out = self.ac_conv(conv(ss).squeeze(3))    
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) # (B, 2)
                ll1.append(pool_out)
            ss1 = torch.cat(ll1, 1).unsqueeze(1) 
            ll2.append(ss1)
            ll1 = []

        return torch.cat(ll2, 1)         # (B, C, 10)      >>>>>>>> Item candi
        '''

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


        return item_repr


    
    def predict(self, interaction):

        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        item_var = self.item_doc.unsqueeze(0)
        
        item_repr = self._get_item_repr(item_var)
        item_repr = item_repr.expand(seq_var.shape[0],-1,-1)  # expand batch     
    
        scores = self.forward(seq_var, user_var, item_repr)   # [B n_items]
        
        return scores

    # get score
    def full_sort_predict(self, interaction):


        seq_var  = self.item_doc[interaction['item_id_list']]
        user_var = self.user_doc[interaction['user_id']] 

        item_var = self.item_doc.unsqueeze(0)
        
        item_repr = self._get_item_repr(item_var)
        item_repr = item_repr.repeat(seq_var.shape[0], 1, 1)  # expand batch

        scores = self.forward(seq_var, user_var, item_repr)   # [B n_items]

        return scores