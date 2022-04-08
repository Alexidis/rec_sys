from implicit.nearest_neighbours import ItemItemRecommender
import pandas as pd
import numpy as np


def get_similar_users_recommendation(user_ids, _id_to_item_value, _user_value_to_id, _id_to_user_value, _sparse_uim,
                                     _f_items, _als_model, n=5):
    
    ii_model = ItemItemRecommender(K=1, num_threads=4)
    ii_model.fit(_sparse_uim, show_progress=True)

    sim_user_df = pd.DataFrame()
    sim_user_df['user_id'] = user_ids
    sim_user_df['own_recommend'] = sim_user_df['user_id'].apply(lambda x:
                                                                [_id_to_item_value[ids]
                                                                 for ids in ii_model.recommend(
                                                                    userid=_user_value_to_id[x],
                                                                    user_items=_sparse_uim[_user_value_to_id[x]],
                                                                    N=n,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[_f_items, ],
                                                                    recalculate_user=True)[0]])

    sim_user_df['sim_user'] = \
        sim_user_df['user_id'].apply(lambda x: _als_model.similar_users(_user_value_to_id[x], 3)[0][1:])

    sim_user_df['sim_user_items_recommends'] = \
        sim_user_df['sim_user'].apply(lambda x:
                                      [_id_to_item_value[ids] for ids in
                                       np.unique([rec_ids for rec_ids in
                                                  _als_model.recommend(
                                                      userid=x,
                                                      user_items=_sparse_uim[x],
                                                      N=n,
                                                      filter_already_liked_items=False,
                                                      filter_items=_f_items,
                                                      recalculate_user=True)[0]]).flatten()[:n+1]])

    return sim_user_df[['user_id', 'sim_user_items_recommends']]


def get_similar_item(model, item_id, _id_to_item_value, _item_value_to_id, _f_items, n=5):
    """Находит товар, похожий на item_id"""
    cur_recs = [_item_value_to_id[ids] for ids in item_id if id != _id_to_item_value[_f_items]]
    sim_item_rec = np.unique([_id_to_item_value[sim[1]] for sim
                              in model.similar_items(cur_recs, N=2, filter_items=[_f_items, ])[0]])[:n]

    return sim_item_rec
