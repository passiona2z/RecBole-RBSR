from recbole.quick_start import run_recbole
import argparse


# RNS (or) PIRSP
rns_parameter_dict = {
   'gpu_id': '0', 

   'load_col': {
      'inter': ['user_id', 'item_id', 'timestamp', 'review']
    },

   'REVIEW_SUFFIX': '_aggreview',

   'train_neg_sample_args': None,

   'epochs': 50,
   'train_batch_size': 128,
   'eval_batch_size': 128,

   'MAX_ITEM_LIST_LENGTH': 5,

   'user_inter_num_interval': "[5,inf)",
   'item_inter_num_interval': "[5,inf)",

   'metrics': ['Hit', 'NDCG'],
   'topk': [5, 10],
   'valid_metric': 'NDCG@10'
}

# CCA
cca_parameter_dict = {
   'gpu_id': '0', 

   'load_col': {
      'inter': ['user_id', 'item_id', 'timestamp', 'review']
    },

   'train_neg_sample_args': None,

   'epochs': 50,
   'train_batch_size': 64,
   'eval_batch_size': 64,

   'MAX_ITEM_LIST_LENGTH': 20,

   'user_inter_num_interval': "[5,inf)",
   'item_inter_num_interval': "[5,inf)",

   'metrics': ['Hit', 'NDCG'],
   'topk': [5, 10],
   'valid_metric': 'NDCG@10'
}

# IntentRec
IntentRec_parameter_dict = {
   'gpu_id': '0', 

   'load_col': {
      'inter': ['user_id', 'item_id', 'timestamp', 'review']
    },


   'train_neg_sample_args': None,

   'epochs': 50,
   'train_batch_size': 128,
   'eval_batch_size': 128,

   'MAX_ITEM_LIST_LENGTH': 20,

   'user_inter_num_interval': "[5,inf)",
   'item_inter_num_interval': "[5,inf)",

   'metrics': ['Hit', 'NDCG'],
   'topk': [5, 10],
   'valid_metric': 'NDCG@10'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='IntentRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='MI', help='name of datasets')

    args, _ = parser.parse_known_args()

    config_dicts = {
        'RNS': rns_parameter_dict,         
        'PIRSP': rns_parameter_dict,       
        'CCA': cca_parameter_dict,        
        'IntentRec': IntentRec_parameter_dict,   
    }

    config_dict = config_dicts.get(args.model, "rns_parameter_dict")

    run_recbole(model=args.model, dataset=args.dataset, config_dict=config_dict)