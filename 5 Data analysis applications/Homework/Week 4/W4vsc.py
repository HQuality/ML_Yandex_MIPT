#%% [markdown]
# # Recommender systems
#%% [markdown]
# ***Описание задачи***
# 
# Небольшой интернет-магазин попросил вас добавить ранжирование товаров в блок "Смотрели ранее" - в нём теперь надо показывать не последние просмотренные пользователем товары, а те товары из просмотренных, которые он наиболее вероятно купит. Качество вашего решения будет оцениваться по количеству покупок в сравнении с прошлым решением в ходе А/В теста, т.к. по доходу от продаж статзначимость будет достигаться дольше из-за разброса цен. Таким образом, ничего заранее не зная про корреляцию оффлайновых и онлайновых метрик качества, в начале проекта вы можете лишь постараться оптимизировать recall@k и precision@k.
# 
# Это задание посвящено построению простых бейзлайнов для этой задачи: ранжирование просмотренных товаров по частоте просмотров и по частоте покупок. Эти бейзлайны, с одной стороны, могут помочь вам грубо оценить возможный эффект от ранжирования товаров в блоке - например, чтобы вписать какие-то числа в коммерческое предложение заказчику, а с другой стороны, могут оказаться самым хорошим вариантом, если данных очень мало (недостаточно для обучения даже простых моделей).
#%% [markdown]
# ***Входные данные***
# 
# Вам дается две выборки с пользовательскими сессиями - id-шниками просмотренных и id-шниками купленных товаров. Одна выборка будет использоваться для обучения (оценки популярностей товаров), а другая - для теста.
# 
# В файлах записаны сессии по одной в каждой строке. Формат сессии: id просмотренных товаров через , затем идёт ; после чего следуют id купленных товаров (если такие имеются), разделённые запятой. Например, 1,2,3,4; или 1,2,3,4;5,6.
# 
# Гарантируется, что среди id купленных товаров все различные.
#%% [markdown]
# ***Важно:***
# 
# Сессии, в которых пользователь ничего не купил, исключаем из оценки качества.
# Если товар не встречался в обучающей выборке, его популярность равна 0.
# Рекомендуем разные товары. И их число должно быть не больше, чем количество различных просмотренных пользователем товаров.
# Рекомендаций всегда не больше, чем минимум из двух чисел: количество просмотренных пользователем товаров и k в recall@k / precision@k.
#%% [markdown]
# ***Задание***
# 
# 1. На обучении постройте частоты появления id в просмотренных и в купленных (id может несколько раз появляться в просмотренных, все появления надо учитывать)
# 2. Реализуйте два алгоритма рекомендаций:
#     * сортировка просмотренных id по популярности (частота появления в просмотренных),
#     * сортировка просмотренных id по покупаемости (частота появления в покупках).
# 3. Для данных алгоритмов выпишите через пробел AverageRecall@1, AveragePrecision@1, AverageRecall@5, AveragePrecision@5 на обучающей и тестовых выборках, округляя до 2 знака после запятой. Это будут ваши ответы в этом задании. Посмотрите, как они соотносятся друг с другом. Где качество получилось выше? Значимо ли это различие? Обратите внимание на различие качества на обучающей и тестовой выборке в случае рекомендаций по частотам покупки.
# 
# Если частота одинаковая, то сортировать нужно по возрастанию момента просмотра (чем раньше появился в просмотренных, тем больше приоритет)

#%%
from __future__ import division, print_function

import numpy as np
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%% [markdown]
# #### 1. Reading sessions train and test datasets.

#%%
# Reading train and test data
with open('coursera_sessions_train.txt', 'r') as file:
    sess_train = file.read().splitlines()
with open('coursera_sessions_test.txt', 'r') as file:
    sess_test = file.read().splitlines()

#%% [markdown]
# #### 2. Split datasets by looks and purchases.

#%%
# Create train array splitted by looks (look_items) and purchases (pur_items)
sess_train_lp = []
for sess in sess_train:
    look_items, pur_items = sess.split(';')
    look_items = map(int, look_items.split(','))
    if len(pur_items) > 0:
        pur_items = map(int, pur_items.split(','))
    else:
        pur_items = []
    sess_train_lp.append([look_items, pur_items])
    
# Create test array splitted by looks (look_items) and purchases (pur_items)
sess_test_lp = []
for sess in sess_test:
    look_items, pur_items = sess.split(';')
    look_items = map(int, look_items.split(','))
    if len(pur_items) > 0:
        pur_items = map(int, pur_items.split(','))
    else:
        pur_items = []
    sess_test_lp.append([look_items, pur_items])

#%% [markdown]
# #### 3. Create and sort arrays of unique ids counters for looks and purchases for train dataset.

#%%
# Array of looks
sess_train_l = [row[0] for row in sess_train_lp]
sess_train_l_np = np.array( [id_n for sess in sess_train_l for id_n in sess] )

# Array of unique ids and looks in train data
sess_train_l_cnt = np.transpose(np.unique(sess_train_l_np, return_counts=True))


#%%
sess_train_l_cnt


#%%
# Array of purchases
sess_train_p = [row[1] for row in sess_train_lp]
sess_train_p_np = np.array( [id_n for sess in sess_train_p for id_n in sess] )

# Array of unique ids and purchases in train dataset
sess_train_p_cnt = np.transpose(np.unique(sess_train_p_np, return_counts=True))


#%%
sess_train_p_cnt


#%%
# Sorting arrays of looks and purchases by counts
sess_train_l_cnt = sess_train_l_cnt[sess_train_l_cnt[:,1].argsort()][::-1]
sess_train_p_cnt = sess_train_p_cnt[sess_train_p_cnt[:,1].argsort()][::-1]

#%% [markdown]
# #### 4. Calculating metrics for train dataset with suggestions based on looks.

#%%
def prec_rec_metrics(session, reccomendations, k):
    purchase = 0
    for ind in reccomendations:
        if ind in session:
            purchase += 1 
    precision = purchase / k
    recall = purchase / len(session)
    return(precision, recall)


#%%
# Calculate metrics for train dataset, suggestions based on looks
prec_at_1_tr_l, rec_at_1_tr_l = [], []
prec_at_5_tr_l, rec_at_5_tr_l = [], []
k1, k5 = 1, 5
for i, sess_p in enumerate(sess_train_p):
    # skip sessions without purchases
    if sess_p == []: continue
    
    # looks ids
    sess_l = sess_train_l[i]

    # sorted looks ids indices in sess_train_l_cnt array
    # sort in accordance with looks counts
    l_ind_sess = []
    for j in range(len(list(sess_l))):
        l_ind_sess.append(np.where(sess_train_l_cnt[:,0] == sess_l[j])[0][0])
    l_ind_sess_sorted = np.unique(l_ind_sess)
    
    # k1 recommendations
    num_of_recs_k1 = min(k1, len(list(sess_l)))
    if num_of_recs_k1 == 0: continue
    recs_k1 = sess_train_l_cnt[l_ind_sess_sorted[:num_of_recs_k1],0]
    
    # k1 metrics
    prec_1, rec_1 = prec_rec_metrics(sess_p, recs_k1, k1)
    prec_at_1_tr_l.append(prec_1)
    rec_at_1_tr_l.append(rec_1)
    
    # k5 recommendations
    num_of_recs_k5 = min(k5, len(list(sess_l)))
    if num_of_recs_k5 == 0: continue
    recs_k5 = sess_train_l_cnt[l_ind_sess_sorted[:num_of_recs_k5],0]
    
    # k5 metrics
    prec_5, rec_5 = prec_rec_metrics(sess_p, recs_k5, k5)
    prec_at_5_tr_l.append(prec_5)
    rec_at_5_tr_l.append(rec_5)


#%%
avg_prec_at_1_tr_l = np.mean(prec_at_1_tr_l)
avg_rec_at_1_tr_l = np.mean(rec_at_1_tr_l)
avg_prec_at_5_tr_l = np.mean(prec_at_5_tr_l)
avg_rec_at_5_tr_l = np.mean(rec_at_5_tr_l)


#%%
with open('ans1.txt', 'w') as file:
    r1 = '%.2f' % round(avg_rec_at_1_tr_l, 2)
    p1 = '%.2f' % round(avg_prec_at_1_tr_l, 2)
    r5 = '%.2f' % round(avg_rec_at_5_tr_l, 2)
    p5 = '%.2f' % round(avg_prec_at_5_tr_l, 2)
    ans1 = ' '.join([r1, p1, r5, p5])
    print('Answer 1:', ans1)
    file.write(ans1)

#%% [markdown]
# #### 5. Calculating metrics for train dataset with suggestions based on purchases.

#%%
# Calculate metrics for train dataset, suggestions based on purchases
prec_at_1_tr_p, rec_at_1_tr_p = [], []
prec_at_5_tr_p, rec_at_5_tr_p = [], []
k1, k5 = 1, 5

for i, sess_p in enumerate(sess_train_p):
    # skip sessions without purchases
    if sess_p == []: continue
    
    # looks ids
    sess_l = sess_train_l[i]

    # sorted looks ids indices in sess_train_p_cnt array
    # sort in accordance with purchases counts
    l_ind_sess = []
    for j in range(len(sess_l)):
        if sess_l[j] not in sess_train_p_cnt[:,0]: continue
        l_ind_sess.append(np.where(sess_train_p_cnt[:,0] == sess_l[j])[0][0])
    l_ind_sess_sorted = np.unique(l_ind_sess)
    
    # k1 recommendations
    num_of_recs_k1 = min(k1, len(sess_l), len(l_ind_sess_sorted))
    if num_of_recs_k1 == 0: continue
    recs_k1 = sess_train_p_cnt[l_ind_sess_sorted[:num_of_recs_k1],0]
    
    # k1 metrics
    prec_1, rec_1 = prec_rec_metrics(sess_p, recs_k1, k1)
    prec_at_1_tr_p.append(prec_1)
    rec_at_1_tr_p.append(rec_1)
    
    # k5 recommendations
    num_of_recs_k5 = min(k5, len(sess_l), len(l_ind_sess_sorted))
    if num_of_recs_k5 == 0: continue
    recs_k5 = sess_train_p_cnt[l_ind_sess_sorted[:num_of_recs_k5],0]
    
    # k5 metrics
    prec_5, rec_5 = prec_rec_metrics(sess_p, recs_k5, k5)
    prec_at_5_tr_p.append(prec_5)
    rec_at_5_tr_p.append(rec_5)


#%%
avg_prec_at_1_tr_p = np.mean(prec_at_1_tr_p)
avg_rec_at_1_tr_p = np.mean(rec_at_1_tr_p)
avg_prec_at_5_tr_p = np.mean(prec_at_5_tr_p)
avg_rec_at_5_tr_p = np.mean(rec_at_5_tr_p)


#%%
with open('ans2.txt', 'w') as file:
    r1 = '%.2f' % round(avg_rec_at_1_tr_p, 2)
    p1 = '%.2f' % round(avg_prec_at_1_tr_p, 2)
    r5 = '%.2f' % round(avg_rec_at_5_tr_p, 2)
    p5 = '%.2f' % round(avg_prec_at_5_tr_p, 2)
    ans2 = ' '.join([r1, p1, r5, p5])
    print('Answer 2:', ans2)
    file.write(ans2)

#%% [markdown]
# #### 6. Create and sort arrays of unique ids counters for looks and purchases for test dataset.

#%%
# Array of looks
sess_test_l = [row[0] for row in sess_test_lp]
sess_test_l_np = np.array( [id_n for sess in sess_test_l for id_n in sess] )

# Array of unique ids and looks in train data
#sess_test_l_cnt = np.transpose(np.unique(sess_test_l_np, return_counts=True))


#%%
sess_test_l_np
#sess_test_l_cnt


#%%
# Array of purchases
sess_test_p = [row[1] for row in sess_test_lp]
sess_test_p_np = np.array( [id_n for sess in sess_test_p for id_n in sess] )

# Array of unique ids and purchases in train dataset
#sess_test_p_cnt = np.transpose(np.unique(sess_test_p_np, return_counts=True))


#%%
sess_test_p_np
#sess_test_p_cnt


#%%
# Sorting arrays of looks and purchases by counts
#sess_train_l_cnt = sess_train_l_cnt[sess_train_l_cnt[:,1].argsort()][::-1]
#sess_train_p_cnt = sess_train_p_cnt[sess_train_p_cnt[:,1].argsort()][::-1]

#%% [markdown]
# #### 7. Calculating metrics for test dataset with suggestions based on looks.

#%%
# Calculate metrics for test dataset, suggestions based on looks
prec_at_1_tst_l, rec_at_1_tst_l = [], []
prec_at_5_tst_l, rec_at_5_tst_l = [], []
k1, k5 = 1, 5

for i, sess_p in enumerate(sess_test_p):
    # skip sessions without purchases
    if sess_p == []: continue
    
    # looks ids
    sess_l = sess_test_l[i]

    # sorted looks ids indices in sess_train_l_cnt array
    # sort in accordance with looks counts
    l_ind_sess = []
    new_ids = []
    for j in range(len(sess_l)):
        if sess_l[j] not in sess_train_l_cnt[:,0]:
            new_ids.append(sess_l[j])
            continue
        l_ind_sess.append(np.where(sess_train_l_cnt[:,0] == sess_l[j])[0][0])
    l_ind_sess_sorted = np.unique(l_ind_sess)
    
    # k1 recommendations
    num_of_recs_k1 = min(k1, len(sess_l))
    if num_of_recs_k1 == 0: continue
    if l_ind_sess != []:
        recs_k1 = sess_train_l_cnt[l_ind_sess_sorted[:num_of_recs_k1],0]
    else:
        recs_k1 = []
    recs_k1 = np.concatenate((np.array(recs_k1, dtype='int64'), np.unique(np.array(new_ids, dtype='int64'))))[:num_of_recs_k1]
    #recs_k1
    
    # k1 metrics
    prec_1, rec_1 = prec_rec_metrics(sess_p, recs_k1, k1)
    prec_at_1_tst_l.append(prec_1)
    rec_at_1_tst_l.append(rec_1)
    
    # k5 recommendations
    num_of_recs_k5 = min(k5, len(sess_l))
    if num_of_recs_k5 == 0: continue
    if l_ind_sess != []:
        recs_k5 = sess_train_l_cnt[l_ind_sess_sorted[:num_of_recs_k5],0]
    else:
        recs_k5 = []
    recs_k5 = np.concatenate((np.array(recs_k5, dtype='int64'), np.unique(np.array(new_ids, dtype='int64'))))[:num_of_recs_k5]
    #recs_k5
    
    # k5 metrics
    prec_5, rec_5 = prec_rec_metrics(sess_p, recs_k5, k5)
    prec_at_5_tst_l.append(prec_5)
    rec_at_5_tst_l.append(rec_5)


#%%
avg_prec_at_1_tst_l = np.mean(prec_at_1_tst_l)
avg_rec_at_1_tst_l = np.mean(rec_at_1_tst_l)
avg_prec_at_5_tst_l = np.mean(prec_at_5_tst_l)
avg_rec_at_5_tst_l = np.mean(rec_at_5_tst_l)


#%%
with open('ans3.txt', 'w') as file:
    r1 = '%.2f' % round(avg_rec_at_1_tst_l, 2)
    p1 = '%.2f' % round(avg_prec_at_1_tst_l, 2)
    r5 = '%.2f' % round(avg_rec_at_5_tst_l, 2)
    p5 = '%.2f' % round(avg_prec_at_5_tst_l, 2)
    ans3 = ' '.join([r1, p1, r5, p5])
    print('Answer 3:', ans3)
    file.write(ans3)

#%% [markdown]
# #### 8. Calculating metrics for test dataset with suggestions based on purchases.

#%%
def uniquifier(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


#%%
# Calculate metrics for test dataset, suggestions based on purchases
prec_at_1_tst_p, rec_at_1_tst_p = [], []
prec_at_5_tst_p, rec_at_5_tst_p = [], []
k1, k5 = 1, 5
for i, sess_p in enumerate(sess_test_p):
    # skip sessions without purchases
    if sess_p == []: continue
    
    # looks ids
    sess_l = sess_test_l[i]

    # sorted looks ids indices in sess_train_p_cnt array
    # sort in accordance with purchases counts
    l_ind_sess = []
    new_ids = []
    for j in range(len(sess_l)):
        if sess_l[j] not in sess_train_p_cnt[:,0]:
            new_ids.append(sess_l[j])
            continue
        l_ind_sess.append(np.where(sess_train_p_cnt[:,0] == sess_l[j])[0][0])
    l_ind_sess_sorted = np.unique(l_ind_sess)
    
    # k1 recommendations
    num_of_recs_k1 = min(k1, len(sess_l))
    if num_of_recs_k1 == 0: continue
    if l_ind_sess != []:
        recs_k1 = sess_train_p_cnt[l_ind_sess_sorted[:num_of_recs_k1],0]
    else:
        recs_k1 = []
    recs_k1 = np.concatenate((np.array(recs_k1, dtype='int64'), np.array(uniquifier(np.array(new_ids, dtype='int64')))))[:num_of_recs_k1]
    
    # k1 metrics
    prec_1, rec_1 = prec_rec_metrics(sess_p, recs_k1, k1)
    prec_at_1_tst_p.append(prec_1)
    rec_at_1_tst_p.append(rec_1)
    
    # k5 recommendations
    num_of_recs_k5 = min(k5, len(sess_l))
    if num_of_recs_k5 == 0: continue
    if l_ind_sess != []:
        recs_k5 = sess_train_p_cnt[l_ind_sess_sorted[:num_of_recs_k5],0]
    else:
        recs_k5 = []
    recs_k5 = np.concatenate((np.array(recs_k5, dtype='int64'), np.array(uniquifier(np.array(new_ids, dtype='int64')))))[:num_of_recs_k5]
    
    # k5 metrics
    prec_5, rec_5 = prec_rec_metrics(sess_p, recs_k5, k5)
    prec_at_5_tst_p.append(prec_5)
    rec_at_5_tst_p.append(rec_5)


#%%
avg_prec_at_1_tst_p = np.mean(prec_at_1_tst_p)
avg_rec_at_1_tst_p = np.mean(rec_at_1_tst_p)
avg_prec_at_5_tst_p = np.mean(prec_at_5_tst_p)
avg_rec_at_5_tst_p = np.mean(rec_at_5_tst_p)


#%%
with open('ans4.txt', 'w') as file:
    r1 = '%.2f' % round(avg_rec_at_1_tst_p, 2)
    p1 = '%.2f' % round(avg_prec_at_1_tst_p, 2)
    r5 = '%.2f' % round(avg_rec_at_5_tst_p, 2)
    p5 = '%.2f' % round(avg_prec_at_5_tst_p, 2)
    ans4 = ' '.join([r1, p1, r5, p5])
    print('Answer 4:', ans4)
    file.write(ans4)


