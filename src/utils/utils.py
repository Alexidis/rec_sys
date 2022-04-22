import numpy as np


def id_converter(ui_matrix):
    userids = ui_matrix.index.values
    itemids = ui_matrix.columns.values
    
    matrix_u_ids = np.arange(len(userids))
    matrix_i_ids = np.arange(len(itemids))
    
    id_to_itemid = dict(zip(matrix_i_ids, itemids))
    id_to_userid = dict(zip(matrix_u_ids, userids))
    
    itemid_to_id = dict(zip(itemids, matrix_i_ids))
    userid_to_id = dict(zip(userids, matrix_u_ids))
    
    return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id


def get_recommends(user, user_item, filter_items, model, n=5):
    ids, scores = model.recommend(userid=user,
                                  user_items=user_item,  # на вход user-item matrix
                                  N=n,
                                  filter_already_liked_items=False,
                                  filter_items=[filter_items, ],
                                  recalculate_user=True)
    return ids


def prefilter_items(df, item_features, cur_week, n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    unq_user_id = df['user_id'].nunique()
    popularity = (df.groupby('item_id')['user_id'].nunique() / unq_user_id).reset_index()
    popularity.columns = ['item_id', 'share_unique_users']
    
    # больше половины купили товар
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    df = df[~df['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    # только 1% процент купил товар
    top_not_popular = popularity[popularity['share_unique_users'] < 0.001].item_id.tolist()
    df = df[~df['item_id'].isin(top_not_popular)]
    
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = (item_features.groupby('department')['item_id'].nunique()).reset_index()
        heavy_department = item_features['department'].isin(['GROCERY', 'DRUG GM'])
        common_items_count = item_features[~heavy_department]['item_id'].nunique()
        department_size['bye_coef'] = department_size['item_id'] / common_items_count
        rare_departments = department_size[department_size['bye_coef'] < 0.001].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()
        df = df[~df['item_id'].isin(items_in_rare_departments)]
    
    df['price'] = df['sales_value'] / (np.maximum(df['quantity'], 1))
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    cheap = df['price'] > 2
    # Уберем слишком дорогие товары
    expensive = df['price'] < 16
    # Уберем не купленные
    not_purchased = df['quantity'] > 0
    df = df[cheap & expensive & not_purchased]
    
    # Возьмем топ по популярности
    popularity = df.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    
    top = popularity.sort_values('n_sold', ascending=False).head(n_popular).item_id.tolist()
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    df.loc[~df['item_id'].isin(top), 'item_id'] = -1
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    df = df[df['week_no'] >= (cur_week - 52)]
    
    return df


def postfilter_items(user_id, recommednations):
    pass

