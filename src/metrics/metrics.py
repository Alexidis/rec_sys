import numpy as np


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def recall_at_k(recommended_list, bought_list, k=5):
    # количество купленных из k рекомендованных / количество купленных
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    flags = np.isin(recommended_list, bought_list)
    recall = flags.sum() / len(bought_list)
    
    return recall
