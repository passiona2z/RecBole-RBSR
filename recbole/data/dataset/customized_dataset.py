# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2025/7/13
# @Author : Seonjin Hwang

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.
Customized datasets named ``[Model Name]Dataset`` can be automatically called.

Build (Update) Customized datasets for RBSR 

"""

import numpy as np
import pandas as pd
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset, Dataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler

from recbole.utils import (
    FeatureSource,
    FeatureType,
    get_local_time,
    set_color,
    ensure_dir,
)

# for RBSR 
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel 

from tqdm import tqdm
tqdm.pandas()
from logging import getLogger

import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


# for IntentRec
class IntentRecDataset(SequentialDataset):

    def __init__(self, config):
        
        super().__init__(config)
        self.logger = getLogger()

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(
            set_color(
                f"Loading feature from [{filepath}] (source: [{source}]).", "green"
            )
        )

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config["field_separator"]
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config["encoding"]
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f"Type {ftype} from field {field} is not supported.")
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != "link":
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith("seq"):
                    self.field2seqlen[field] = 1
                if "float" in ftype.value:
                    self.field2bucketnum[field] = 2
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f"No columns has been loaded from [{source}]")
            return None

        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            engine="python",
        )
        df.columns = columns

        seq_separator = self.config["seq_separator"]
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith("seq"):
                continue
            df[field].fillna(value="", inplace=True)


            if ftype == FeatureType.TOKEN_SEQ and field == 'review' :
                
                self.logger.debug(f'review_tokenizing')
                df = self.review_tokenizing(df)
            
            elif ftype == FeatureType.TOKEN_SEQ:
                df[field] = [
                    np.array(list(filter(None, _.split(seq_separator))))
                    for _ in df[field].values
                ]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [
                    np.array(list(map(float, filter(None, _.split(seq_separator)))))
                    for _ in df[field].values
                ]
            max_seq_len = max(map(len, df[field].values))
            if self.config["seq_len"] and field in self.config["seq_len"]:
                seq_len = self.config["seq_len"][field]
                df[field] = [
                    seq[:seq_len] if len(seq) > seq_len else seq
                    for seq in df[field].values
                ]
                self.field2seqlen[field] = min(seq_len, max_seq_len)
            else:
                self.field2seqlen[field] = max_seq_len

        return df
        
    # perform in load_feat : review text > review token
    def review_tokenizing(self, df) :
    
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]  

        review_array = df['review'].values

        self.logger.info(f'    tokenizer.encode')

        # tqdm
        li = [self.tokenizer.encode(review, add_special_tokens=True, max_length=256, padding='max_length', truncation=True) for review in tqdm(review_array)]

        df['review'] = li

        return df


    
    def _remap_ID_all(self):
        """Remap all token-like fields.
           review : no remap
        """

        self.logger.debug(f'self.alias.values() {self.alias.values()}')
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        # review
        self.logger.debug(f'self._rest_fields {self._rest_fields}')
        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            # self.logger.debug(f'_rest_fields_remap_list {remap_list}')
            if field == 'review' :
                continue
            else : self._remap(remap_list)


    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        
        self.get_cls()
        # https://stackoverflow.com/questions/31232098/how-to-call-super-method-from-grandchild-class
        Dataset._change_feat_format(self)  # not call SequentialDataset
        
        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")

        self.data_augmentation()


    # gpu processing
    def get_cls(self) :


        setattr(self, f"cls_field", "init_cls")  # attribute creation

        # ftype : FeatureType.TOKEN_SEQ
        self.set_field_property(
            "init_cls", FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION, 768)
        
    
        self.logger.info(f"@ get_cls")
        
        self.model_name ='bert-base-uncased'
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased', add_pooling_layer=False).to(self.config.device)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        reviews = torch.LongTensor(self.inter_feat['review'].to_list()).to(self.config.device)
        
        # check folder
        folder_path = f'dataset/CLS'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # check file
        cls_path = f'{folder_path}/{self.dataset_name}_{self.model_name}.pt'
        
        if os.path.exists(cls_path):
            self.logger.debug(f"    : load from {cls_path}")
            cls_data = torch.load(cls_path)
            
        else :     
            self.text_encoder.eval()
            with torch.no_grad():
            
                li = []    
                
                for review in tqdm(reviews.split(64)) :  # default 64
                    att_mask = (review > 0).long()
                    out_cls = self.text_encoder(input_ids= review, attention_mask= att_mask).last_hidden_state[:,0,:]
                    
                    copy = out_cls.cpu().detach()
                    li.append(copy)
                    
                cls_data = torch.cat(li)
                torch.save(cls_data, cls_path)
                self.logger.debug(f"    : save {cls_path}")

                del review, att_mask, out_cls
                
        # https://stackoverflow.com/questions/55788093/how-to-free-gpu-memory-by-deleting-tensors
        del self.text_encoder, reviews


        self.inter_feat['init_cls'] = [i for i in cls_data.numpy()]

# for CCA
class CCADataset(SequentialDataset):

    def __init__(self, config):
        
        super().__init__(config)
        self.logger = getLogger()

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(
            set_color(
                f"Loading feature from [{filepath}] (source: [{source}]).", "green"
            )
        )

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config["field_separator"]
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config["encoding"]
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f"Type {ftype} from field {field} is not supported.")
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != "link":
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith("seq"):
                    self.field2seqlen[field] = 1
                if "float" in ftype.value:
                    self.field2bucketnum[field] = 2
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f"No columns has been loaded from [{source}]")
            return None

        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            engine="python",
        )
        df.columns = columns

        seq_separator = self.config["seq_separator"]
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith("seq"):
                continue
            df[field].fillna(value="", inplace=True)

            # first review processing
            if ftype == FeatureType.TOKEN_SEQ and field == 'review' :
                
                self.logger.debug(f'review_tokenizing')
                df = self.review_tokenizing(df)
            
            elif ftype == FeatureType.TOKEN_SEQ:
                df[field] = [
                    np.array(list(filter(None, _.split(seq_separator))))
                    for _ in df[field].values
                ]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [
                    np.array(list(map(float, filter(None, _.split(seq_separator)))))
                    for _ in df[field].values
                ]
            max_seq_len = max(map(len, df[field].values))
            if self.config["seq_len"] and field in self.config["seq_len"]:
                seq_len = self.config["seq_len"][field]
                df[field] = [
                    seq[:seq_len] if len(seq) > seq_len else seq
                    for seq in df[field].values
                ]
                self.field2seqlen[field] = min(seq_len, max_seq_len)
            else:
                self.field2seqlen[field] = max_seq_len

        return df
        
    # review text > review token
    def review_tokenizing(self, df) :
    
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        review_array = df['review'].values

        self.logger.info(f'    tokenizer.encode')

        # tqdm
        li = [self.tokenizer.encode(review, add_special_tokens=True, max_length=256, padding='max_length', truncation=True) for review in tqdm(review_array)]

        df['review'] = li

        return df

    
    def _remap_ID_all(self):
        """Remap all token-like fields.
           review : no remap
        """

        self.logger.debug(f'self.alias.values() {self.alias.values()}')
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        # review
        self.logger.debug(f'self._rest_fields {self._rest_fields}')
        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            # self.logger.debug(f'_rest_fields_remap_list {remap_list}')
            if field == 'review' :
                continue
            else : self._remap(remap_list)

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        
        self.get_cls()

        self.get_aggcls() 

        self.get_ts()
        
        # https://stackoverflow.com/questions/31232098/how-to-call-super-method-from-grandchild-class
        Dataset._change_feat_format(self)  # not call SequentialDataset
        
        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")

        self.data_augmentation()



    def get_ts(self) :



        setattr(self, f"cls_field", "time_emb")  # attribute creation

        # ftype : FeatureType.TOKEN_SEQ
        self.set_field_property(
            "time_emb", FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION, 6)
        
    
        self.logger.debug(f"@ get_ts emb")

        # https://github.com/BING303/CCA/blob/main/processed_data.py#L231
        df_ts = self.inter_feat['timestamp']
        df_ts = pd.to_datetime(df_ts, unit='s')
        
        time_df = pd.DataFrame()    # init
        time_df['year'], time_df['month'], time_df['day'], time_df['dayofweek'], time_df['dayofyear'] , time_df['week'] \
                = zip(*df_ts.map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))


        time_df['year']-=time_df['year'].min()
        time_df['year']/=time_df['year'].max()
        time_df['month']/=12
        time_df['day']/=31
        time_df['dayofweek']/=7
        time_df['dayofyear']/=365
        time_df['week']/=52
        
        time_df.fillna(0,inplace=True)

        time_df.to_numpy()

        self.inter_feat['time_emb'] = [i for i in time_df.to_numpy().round(5)]
    
    # gpu processing
    def get_cls(self) :

        setattr(self, f"cls_field", "init_cls")  # attribute creation

        # ftype : FeatureType.TOKEN_SEQ
        self.set_field_property(
            "init_cls", FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION, 768)
        
    
        self.logger.info(f"@ get_cls")
        
        self.model_name ='bert-base-uncased'
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased', add_pooling_layer=False).to(self.config.device)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        reviews = torch.LongTensor(self.inter_feat['review'].to_list()).to(self.config.device)
        
        # check folder
        folder_path = f'dataset/CLS'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # check file
        cls_path = f'{folder_path}/{self.dataset_name}_{self.model_name}.pt'
        
        if os.path.exists(cls_path):
            self.logger.debug(f"    : load from {cls_path}")
            cls_data = torch.load(cls_path)
            
        else :     
            self.text_encoder.eval()
            with torch.no_grad():
            
                li = []    
                
                for review in tqdm(reviews.split(64)) :  # default 64
                    att_mask = (review > 0).long()
                    out_cls = self.text_encoder(input_ids= review, attention_mask= att_mask).last_hidden_state[:,0,:]
                    
                    copy = out_cls.cpu().detach()
                    li.append(copy)
                    
                cls_data = torch.cat(li)
                torch.save(cls_data, cls_path)
                self.logger.debug(f"    : save {cls_path}")

                del review, att_mask, out_cls
                
        # https://stackoverflow.com/questions/55788093/how-to-free-gpu-memory-by-deleting-tensors
        del self.text_encoder, reviews


        self.inter_feat['init_cls'] = [i for i in cls_data.numpy()]


    # gpu processing
    def get_aggcls(self) :


        setattr(self, f"item_cls_field", "item_cls_mean")  # attribute creation

        # ftype : FeatureType.TOKEN_SEQ
        self.set_field_property(
            "item_cls_mean", FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION, 768)

        self.logger.debug(f"@ get_aggcls")
                         # item_id                                        # index, 768
        self.item_cls_dict = self.inter_feat[[self.iid_field,'init_cls']].groupby(by=self.iid_field).mean().to_dict()['init_cls'] # dict

        # loop
        cls_li = []
        for i in self.inter_feat[self.iid_field] :
            cls_li.append(self.item_cls_dict[i])

        self.inter_feat['item_cls_mean'] = cls_li
        self.item_cls_dict.update(({0: np.array([0]*768)})) # 0 : pad item
        
        
class RNSDataset(SequentialDataset):

    def __init__(self, config):
        
        super().__init__(config)
        self.logger = getLogger()


    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(
            set_color(
                f"Loading feature from [{filepath}] (source: [{source}]).", "green"
            )
        )

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config["field_separator"]
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config["encoding"]
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f"Type {ftype} from field {field} is not supported.")
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != "link":
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith("seq"):
                    self.field2seqlen[field] = 1
                if "float" in ftype.value:
                    self.field2bucketnum[field] = 2
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f"No columns has been loaded from [{source}]")
            return None

        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            engine="python",
        )
        df.columns = columns

        seq_separator = self.config["seq_separator"]
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith("seq"):
                continue
            df[field].fillna(value="", inplace=True)


            if ftype == FeatureType.TOKEN_SEQ and field == 'review' :

                self.logger.debug(f'review_processing')
                df = self.review_processing(df)
            
            elif ftype == FeatureType.TOKEN_SEQ:
                df[field] = [
                    np.array(list(filter(None, _.split(seq_separator))))
                    for _ in df[field].values
                ]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [
                    np.array(list(map(float, filter(None, _.split(seq_separator)))))
                    for _ in df[field].values
                ]
            max_seq_len = max(map(len, df[field].values))
            if self.config["seq_len"] and field in self.config["seq_len"]:
                seq_len = self.config["seq_len"][field]
                df[field] = [
                    seq[:seq_len] if len(seq) > seq_len else seq
                    for seq in df[field].values
                ]
                self.field2seqlen[field] = min(seq_len, max_seq_len)
            else:
                self.field2seqlen[field] = max_seq_len

        return df
        

    def review_processing(self, df) :
    
        self.logger.info(f'   review - tokenize')
        df['review'] = df['review'].progress_apply(lambda x : self.tokenize(x))
        
        return df

    # for review_processing
    def tokenize(self, document) : 
        
        punctuations = ['^', ')', '!', '`', '|', ',', '=', '+', '"', '>', ':', '.',  \
                        '#', '(', '~', ';', '&', '-', '_', '\\', '<', '[', '/', ']', \
                        '*', '$', '?', '@', '%']
            
        clean_tokens = []

        
        for token in word_tokenize(document.lower()) :
                          
            for p in punctuations :
                token = token.replace(p, '')
    
            if len(token) < 1 : 
                continue
                
            if token not in stopwords.words('english') :
                clean_tokens.append(token)
    
        return clean_tokens
    
    # for review_processing    
    def identity_tokenizer(self, text):
        return text
    
    # for review_processing        
    def check_word(self, document, importance_word_list) : 
    
        word_list = []
    
        importance_word_list = set(importance_word_list)
        
        for word in document :
            if word in importance_word_list :
                word_list.append(word)
    
        return word_list
    

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """

        self.logger.debug("get_aggreview.")
        self.get_aggreview()
        
        super()._change_feat_format()  # call SequentialDataset

  
    
    def get_aggreview(self) :
    
        review_suffix = self.config["REVIEW_SUFFIX"] # _aggreview


        for field in self.inter_feat:

            # user_aggreview
            if field in [self.uid_field, self.iid_field] :
                
                review_field = field + review_suffix   # user_id_aggreview, item_id_aggreview
                setattr(self, f"{field}_review_field", review_field)  # attribute creation
                ftype = self.field2type[field]
    
                # ftype : FeatureType.TOKEN_SEQ
                self.set_field_property(
                    review_field, FeatureType.TOKEN_SEQ, FeatureSource.INTERACTION, 100)

                    
        drop_index_set = self._get_review_user(self.uid_field, self.iid_field)
        # self._get_review_item(self.iid_field, self.uid_field, drop_index_set) 
        self._get_review_item(self.iid_field, self.uid_field) 
    
    
    def _aug_presets(self):

        list_suffix   = self.config["LIST_SUFFIX"]


        for field in self.inter_feat:
        
            # if field != self.uid_field:
            if field not in [self.uid_field, self.user_id_review_field]:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)  # attribute creation
                ftype = self.field2type[field]
    
                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ
    
                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len
    
                # new field
                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )
    
            self.set_field_property(
                self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
            )



    def _get_review_item(self, sub, obj, drop_index_set=[], word_length=100) :

        self.logger.debug(f"drop_index_set len : {len(drop_index_set)}")

                          
        df = self.inter_feat.drop(index=drop_index_set).sort_values([sub, self.time_field]).copy()

        reviews_by_sub = dict(list(df[[obj, 'review']].groupby(df[sub])))

        # all_reviews = []
        reviews_dic = {}
        non_review_item = 0
        
        # obj_id
        for idx, (sub_id, obj_id) in enumerate(zip(df[sub], df[obj])):
                
                df_data = reviews_by_sub[sub_id]                                          
            
                reviews = df_data['review'].to_list()                    # [review] * len

                # Exception handling: cases where there is no review dataset for training > padding
                try :
                    reviews = np.concatenate(reviews, axis=0)            # concat

                except : 
                    reviews = np.zeros(1, dtype='int')
                    non_review_item += 1
            
        
                reviews = np.concatenate([reviews[:word_length], \
                                          [0] * (word_length - len(reviews))], axis=0)         # padding for make tensor
            
                if sub_id not in reviews_dic :
                    reviews_dic[sub_id] = reviews.astype(np.int64)

        self.logger.debug(f"non_review_item count : {non_review_item}")
        self.logger.debug(f"{sub} - reviews_dic len : {len(reviews_dic)}")

        self.item_doc = reviews_dic
        self.item_doc.update({0: np.array([0]*word_length)}) # pad
        
        
        dic2series = pd.Series(reviews_dic).reset_index().rename(columns={'index': f'{sub}', 0:f'{sub}_aggreview'})
        self.inter_feat = self.inter_feat.merge(dic2series, how='left', on=sub)

                          
        
    def _get_review_user(self, sub, obj, word_length=100) :
        
        df = self.inter_feat.sort_values([sub, self.time_field]).copy()

        reviews_by_sub = dict(list(df[[obj, 'review']].groupby(df[sub])))

        #all_reviews = []
        reviews_dic = {}
        drop_index_set = set()

        # obj_id 
        for idx, (sub_id, obj_id) in enumerate(zip(df[sub], df[obj])):
                
                df_data = reviews_by_sub[sub_id].iloc[:-1]                                     # not show valid, test

                drop_index = reviews_by_sub[sub_id].iloc[-1:].index.to_list()                  # get_drop_index
        
                drop_index_set.update(drop_index)
            
                reviews = df_data['review'].to_list()                                          # [review] * len
                #self.logger.debug(f"reviews {reviews}")
                reviews = np.concatenate(reviews, axis=0)                                      # concat
                reviews = np.concatenate([reviews[:word_length], \
                                          [0] * (word_length - len(reviews))], axis=0)         # padding for make tensor
            
                if sub_id not in reviews_dic :
                    reviews_dic[sub_id] = reviews.astype(np.int64)        

        self.logger.debug(f"{sub} - reviews_dic len : {len(reviews_dic)}")
        self.user_doc = reviews_dic
        self.user_doc.update({0: np.array([0]*word_length)}) # pad
        
        dic2series = pd.Series(reviews_dic).reset_index().rename(columns={'index': f'{sub}', 0:f'{sub}_aggreview'})

        self.inter_feat = self.inter_feat.merge(dic2series, how='left', on=sub)

        return drop_index_set

    def data_augmentation(self):

        self.logger.debug("   data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")

        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]

        self.sort(by=[self.uid_field, self.time_field], ascending=True)

        
        # 시작        
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        
        seq_start = 0

        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):

            r"""
            origin 1 1 1 2 2 2   
            
            aug 1 1, 1 1 1, 1 1
                2 2, 2 2 2, 2 2 > slice
            """
            
            if last_uid != uid:
                last_uid = uid   
                seq_start = i    


            else:  # last_uid == uid

                if i - seq_start > max_item_list_len:
                    seq_start += 1


                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i)) 
                
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64) # all max len

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

  
        for field in self.inter_feat:

            if field not in [self.uid_field, self.user_id_review_field]:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                    
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        # Interaction class update 
        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data



class PIRSPDataset(RNSDataset):

    def __init__(self, config):
        
        super().__init__(config)
        self.logger = getLogger()

    # perform tfidf in load_feat vocab : 35000
    def review_processing(self, df) :
    
        self.logger.info(f'   review - tokenize')
        df['review'] = df['review'].progress_apply(lambda x : self.tokenize(x))
        
        
        tfidf = TfidfVectorizer(tokenizer=self.identity_tokenizer, lowercase=False)    
        tfidf_separate = tfidf.fit_transform(df['review'])
        
        self.logger.debug(f'   count - tfidf_value')
        word_lst = tfidf.get_feature_names_out()
        count_lst = tfidf_separate.toarray().sum(axis=0)
        vocab_df = pd.DataFrame((zip(word_lst,count_lst)), columns= ["vocab","tfidf_value"])
        
        importance_word_list = vocab_df.sort_values(by="tfidf_value",ascending=False)['vocab'].iloc[:35000].to_list()

        self.logger.debug(f'   filtering_importance_word')
        df['review'] = df['review'].progress_apply(lambda x : self.check_word(x, importance_word_list))
        
        return df


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][:length] = (
                            self.neg_item_list[index]
                        )

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data
