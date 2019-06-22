#%% [markdown]
# <center>
# <img src="https://habrastorage.org/web/677/8e1/337/6778e1337c3d4b159d7e99df94227cb2.jpg"/>
# ## Специализация "Машинное обучение и анализ данных"
# </center>
# <center>Автор материала: программист-исследователь Mail.ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ Юрий Кашницкий
#%% [markdown]
# # <center> Capstone проект №1. Идентификация пользователей по посещенным веб-страницам
# 
# В этом проекте мы будем решать задачу идентификации пользователя по его поведению в сети Интернет. Это сложная и интересная задача на стыке анализа данных и поведенческой психологии. В качестве примера, компания Яндекс решает задачу идентификации взломщика почтового ящика по его поведению. В двух словах, взломщик будет себя вести не так, как владелец ящика: он может не удалять сообщения сразу по прочтении, как это делал хозяин, он будет по-другому ставить флажки сообщениям и даже по-своему двигать мышкой. Тогда такого злоумышленника можно идентифицировать и "выкинуть" из почтового ящика, предложив хозяину войти по SMS-коду. Этот пилотный проект описан в [статье](https://habrahabr.ru/company/yandex/blog/230583/) на Хабрахабре. Похожие вещи делаются, например, в Google Analytics и описываются в научных статьях, найти можно многое по фразам "Traversal Pattern Mining" и "Sequential Pattern Mining".
# 
# <img src='http://i.istockimg.com/file_thumbview_approve/21546327/5/stock-illustration-21546327-identification-de-l-utilisateur.jpg'>
# 
# Мы будем решать похожую задачу: по последовательности из нескольких веб-сайтов, посещенных подряд один и тем же человеком, мы будем идентифицировать этого человека. Идея такая: пользователи Интернета по-разному переходят по ссылкам, и это может помогать их идентифицировать (кто-то сначала в почту, потом про футбол почитать, затем новости, контакт, потом наконец – работать, кто-то – сразу работать).
# 
# Будем использовать данные из [статьи](http://ceur-ws.org/Vol-1703/paper12.pdf) "A Tool for Classification of Sequential Data". И хотя мы не можем рекомендовать эту статью (описанные методы делеки от state-of-the-art, лучше обращаться к [книге](http://www.charuaggarwal.net/freqbook.pdf) "Frequent Pattern Mining" и последним статьям с ICDM), но данные там собраны аккуратно и представляют интерес.
# 
# Имеются данные с прокси-серверов Университета Блеза Паскаля, они имеют очень простой вид. Для каждого пользователя заведен csv-файл с названием user\*\*\*\*.csv (где вместо звездочек – 4 цифры, соответствующие ID пользователя), а в нем посещения сайтов записаны в следующем формате: <br>
# 
# <center>*timestamp, посещенный веб-сайт*</center>
# 
# Скачать исходные данные можно по ссылке в статье, там же описание.
# Для этого задания хватит данных не по всем 3000 пользователям, а по 10 и 150. [Ссылка](https://yadi.sk/d/3gscKIdN3BCASG) на архив *capstone_user_identification* (~7 Mb, в развернутом виде ~ 60 Mb). 
# 
# В финальном проекте уже придется столкнуться с тем, что не все операции можно выполнить за разумное время (скажем, перебрать с кросс-валидацией 100 комбинаций параметров случайного леса на этих данных Вы вряд ли сможете), поэтому мы будем использовать параллельно 2 выборки: по 10 пользователям и по 150. Для 10 пользователей будем писать и отлаживать код, для 150 – будет рабочая версия. 
# 
# Данные устроены следующем образом:
# 
#  - В каталоге 10users лежат 10 csv-файлов с названием вида "user[USER_ID].csv", где [USER_ID] – ID пользователя;
#  - Аналогично для каталога 150users – там 150 файлов;
#  - В каталоге 3users – игрушечный пример из 3 файлов, это для отладки кода предобработки, который Вы далее напишете.
# 
# На 5 неделе будет задание по [соревнованию](https://inclass.kaggle.com/c/identify-me-if-you-can4) Kaggle Inclass, которое организовано специально под Capstone проект нашей специализации. Соревнование уже открыто и, конечно, желающие могут начать уже сейчас.
# 
# # <center>Неделя 1. Подготовка данных к анализу и построению моделей
# 
# Первая часть проекта посвящена подготовке данных для дальнейшего описательного анализа и построения прогнозных моделей. Надо будет написать код для предобработки данных (исходно посещенные веб-сайты указаны для каждого пользователя в отдельном файле) и формирования единой обучающей выборки. Также в этой части мы познакомимся с разреженным форматом данных (матрицы `Scipy.sparse`), который хорошо подходит для данной задачи. 
# 
# **План 1 недели:**
#  - Часть 1. Подготовка обучающей выборки
#  - Часть 2. Работа с разреженным форматом данных
#%% [markdown]
# <font color='red'>**Задание:**</font> заполните код в этой тетрадке и выберите ответы в [веб-форме](https://docs.google.com/forms/d/e/1FAIpQLSedmwHb4cOI32zKJmEP7RvgEjNoz5GbeYRc83qFXVH82KFgGA/viewform). 
# 
#%% [markdown]
# **В этой части проекта Вам могут быть полезны видеозаписи следующих лекций 1 и 2 недели курса "Математика и Python для анализа данных":**
#    - [Циклы, функции, генераторы, list comprehension](https://www.coursera.org/learn/mathematics-and-python/lecture/Kd7dL/tsikly-funktsii-ghienieratory-list-comprehension)
#    - [Чтение данных из файлов](https://www.coursera.org/learn/mathematics-and-python/lecture/8Xvwp/chtieniie-dannykh-iz-failov)
#    - [Запись файлов, изменение файлов](https://www.coursera.org/learn/mathematics-and-python/lecture/vde7k/zapis-failov-izmienieniie-failov)
#    - [Pandas.DataFrame](https://www.coursera.org/learn/mathematics-and-python/lecture/rcjAW/pandas-data-frame)
#    - [Pandas. Индексация и селекция](https://www.coursera.org/learn/mathematics-and-python/lecture/lsXAR/pandas-indieksatsiia-i-sieliektsiia)
#    
# **Кроме того, в задании будут использоваться библиотеки Python [`glob`](https://docs.python.org/3/library/glob.html), [`pickle`](https://docs.python.org/2/library/pickle.html) и класс [`csr_matrix`](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html) из `Scipy.sparse`.**
#%% [markdown]
# Наконец, для лучшей воспроизводимости результатов приведем список версий основных используемых в проекте библиотек: NumPy, SciPy, Pandas, Matplotlib, Statsmodels и Scikit-learn. Для этого воспользуемся расширением [watermark](https://github.com/rasbt/watermark).

#%%
# pip install watermark
get_ipython().run_line_magic('load_ext', 'watermark')


#%%
get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,pandas,matplotlib,statsmodels,sklearn -g')


#%%
from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import os
import pickle
#pip install tqdm
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

#%% [markdown]
# **Посмотрим на один из файлов с данными о посещенных пользователем (номер 31) веб-страницах.**

#%%
# Поменяйте на свой путь к данным
PATH_TO_DATA = '~/capstone_user_identification'


#%%
user31_data = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                       '10users/user0031.csv'))


#%%
user31_data.head()

#%% [markdown]
# **Поставим задачу классификации: идентифицировать пользователя по сессии из 10 подряд посещенных сайтов. Объектом в этой задаче будет сессия из 10 сайтов, последовательно посещенных одним и тем же пользователем, признаками – индексы этих 10 сайтов (чуть позже здесь появится "мешок" сайтов, подход Bag of Words). Целевым классом будет id пользователя.**
#%% [markdown]
# ### <center>Пример для иллюстрации</center>
# **Пусть пользователя всего 2, длина сессии – 2 сайта.**
# 
# <center>user0001.csv</center>
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">timestamp</th>
#     <th class="tg-031e">site</th>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:01</td>
#     <td class="tg-031e">vk.com</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">00:00:11</td>
#     <td class="tg-yw4l">google.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:16</td>
#     <td class="tg-031e">vk.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:20</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
# </table>
# 
# <center>user0002.csv</center>
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">timestamp</th>
#     <th class="tg-031e">site</th>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:02</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">00:00:14</td>
#     <td class="tg-yw4l">google.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:17</td>
#     <td class="tg-031e">facebook.com</td>
#   </tr>
#   <tr>
#     <td class="tg-031e">00:00:25</td>
#     <td class="tg-031e">yandex.ru</td>
#   </tr>
# </table>
# 
# Идем по 1 файлу, нумеруем сайты подряд: vk.com – 1, google.com – 2 и т.д. Далее по второму файлу. 
# 
# Отображение сайтов в их индесы должно получиться таким:
# 
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-yw4l{vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-031e">site</th>
#     <th class="tg-yw4l">site_id</th>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">vk.com</td>
#     <td class="tg-yw4l">1</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">google.com</td>
#     <td class="tg-yw4l">2</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">yandex.ru</td>
#     <td class="tg-yw4l">3</td>
#   </tr>
#   <tr>
#     <td class="tg-yw4l">facebook.com</td>
#     <td class="tg-yw4l">4</td>
#   </tr>
# </table>
# 
# Тогда обучающая выборка будет такой (целевой признак – user_id):
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
# .tg .tg-s6z2{text-align:center}
# .tg .tg-baqh{text-align:center;vertical-align:top}
# .tg .tg-hgcj{font-weight:bold;text-align:center}
# .tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
# </style>
# <table class="tg">
#   <tr>
#     <th class="tg-hgcj">session_id</th>
#     <th class="tg-hgcj">site1</th>
#     <th class="tg-hgcj">site2</th>
#     <th class="tg-amwm">user_id</th>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-baqh">1</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-s6z2">1</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-baqh">1</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-s6z2">2</td>
#     <td class="tg-baqh">2</td>
#   </tr>
#   <tr>
#     <td class="tg-s6z2">4</td>
#     <td class="tg-s6z2">4</td>
#     <td class="tg-s6z2">3</td>
#     <td class="tg-baqh">2</td>
#   </tr>
# </table>
# 
# Здесь 1 объект – это сессия из 2 посещенных сайтов 1-ым пользователем (target=1). Это сайты vk.com и google.com (номер 1 и 2). И так далее, всего 4 сессии. Пока сессии у нас не пересекаются по сайтам, то есть посещение каждого отдельного сайта относится только к одной сессии.
#%% [markdown]
# ## Часть 1. Подготовка обучающей выборки
# Реализуйте функцию *prepare_train_set*, которая принимает на вход путь к каталогу с csv-файлами *path_to_csv_files* и параметр *session_length* – длину сессии, а возвращает 2 объекта:
# - DataFrame, в котором строки соответствуют уникальным сессиям из *session_length* сайтов, *session_length* столбцов – индексам этих *session_length* сайтов и последний столбец – ID пользователя
# - частотный словарь сайтов вида {'site_string': [site_id, site_freq]}, например для недавнего игрушечного примера это будет {'vk.com': (1, 2), 'google.com': (2, 2), 'yandex.ru': (3, 3), 'facebook.com': (4, 1)}
# 
# Детали:
# - Смотрите чуть ниже пример вывода, что должна возвращать функция
# - Используйте `glob` (или аналоги) для обхода файлов в каталоге. Для определенности, отсортируйте список файлов лексикографически. Удобно использовать `tqdm_notebook` (или просто `tqdm` в случае python-скрипта) для отслеживания числа выполненных итераций цикла
# - Создайте частотный словарь уникальных сайтов (вида {'site_string': (site_id, site_freq)}) и заполняйте его по ходу чтения файлов. Начните с 1
# - Рекомендуется меньшие индексы давать более часто попадающимся сайтам (приницип наименьшего описания)
# - Не делайте entity recognition, считайте *google.com*, *http://www.google.com* и *www.google.com* разными сайтами (подключить entity recognition можно уже в рамках индивидуальной работы над проектом)
# - Скорее всего в файле число записей не кратно числу *session_length*. Тогда последняя сессия будет короче. Остаток заполняйте нулями. То есть если в файле 24 записи и сессии длины 10, то 3 сессия будет состоять из 4 сайтов, и ей мы сопоставим вектор [*site1_id*, *site2_id*, *site3_id*, *site4_id*, 0, 0, 0, 0, 0, 0, *user_id*] 
# - В итоге некоторые сессии могут повторяться – оставьте как есть, не удаляйте дубликаты. Если в двух сессиях все сайты одинаковы, но сессии принадлежат разным пользователям, то тоже оставляйте как есть, это естественная неопределенность в данных
# - Не оставляйте в частотном словаре сайт 0 (уже в конце, когда функция возвращает этот словарь)
# - 150 файлов из *capstone_websites_data/150users/* у меня обработались за 1.7 секунды, но многое, конечно, зависит от реализации функции и от используемого железа. И вообще, первая реализация скорее всего будет не самой эффективной, дальше можно заняться профилированием (особенно если планируете запускать этот код для 3000 пользователей). Также эффективная реализация этой функции поможет нам на следующей неделе.

#%%
def prepare_train_set(path_to_csv_files, session_length=10):
    ''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# **Примените полученную функцию к игрушечному примеру, убедитесь, что все работает как надо.**

#%%
get_ipython().system('cat $PATH_TO_DATA/3users/user0001.csv')


#%%
get_ipython().system('cat $PATH_TO_DATA/3users/user0002.csv')


#%%
get_ipython().system('cat $PATH_TO_DATA/3users/user0003.csv')


#%%
train_data_toy, site_freq_3users = prepare_train_set(os.path.join(PATH_TO_DATA, '3users'), 
                                                     session_length=10)


#%%
train_data_toy

#%% [markdown]
# Частоты сайтов (второй элемент кортежа) точно должны быть такими, нумерация может быть любой (первые элементы кортежей могут отличаться).

#%%
site_freq_3users

#%% [markdown]
# Примените полученную функцию к данным по 10 пользователям.
# 
# **<font color='red'> Вопрос 1. </font> Сколько уникальных сессий из 10 сайтов в выборке с 10 пользователями?**

#%%
train_data_10users, site_freq_10users = ''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# **<font color='red'> Вопрос 2. </font> Сколько всего уникальных сайтов в выборке из 10 пользователей? **

#%%
''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# Примените полученную функцию к данным по 150 пользователям.
# 
# **<font color='red'> Вопрос 3. </font> Сколько уникальных сессий из 10 сайтов в выборке с 150 пользователями?**

#%%
get_ipython().run_cell_magic('time', '', "train_data_150users, site_freq_150users = ''' ВАШ КОД ЗДЕСЬ '''")

#%% [markdown]
# **<font color='red'> Вопрос 4. </font> Сколько всего уникальных сайтов в выборке из 150 пользователей? **

#%%
''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# **<font color='red'> Вопрос 5. </font> Какой из этих сайтов НЕ входит в топ-10 самых популярных сайтов среди посещенных 150 пользователями?**
# - www.google.fr
# - www.youtube.com
# - safebrowsing-cache.google.com
# - www.linkedin.com

#%%
''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# **Для дальнейшего анализа запишем полученные объекты DataFrame в csv-файлы.**

#%%
train_data_10users.to_csv(os.path.join(PATH_TO_DATA, 
                                       'train_data_10users.csv'), 
                        index_label='session_id', float_format='%d')
train_data_150users.to_csv(os.path.join(PATH_TO_DATA, 
                                        'train_data_150users.csv'), 
                         index_label='session_id', float_format='%d')

#%% [markdown]
# ## Часть 2. Работа с разреженным форматом данных
#%% [markdown]
# Если так подумать, то полученные признаки *site1*, ..., *site10* смысла не имеют как признаки в задаче классификации. А вот если воспользоваться идеей мешка слов из анализа текстов – это другое дело. Создадим новые матрицы, в которых строкам будут соответствовать сессии из 10 сайтов, а столбцам – индексы сайтов. На пересечении строки $i$ и столбца $j$ будет стоять число $n_{ij}$ – cколько раз сайт $j$ встретился в сессии номер $i$. Делать это будем с помощью разреженных матриц Scipy – [csr_matrix](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html). Прочитайте документацию, разберитесь, как использовать разреженные матрицы и создайте такие матрицы для наших данных. Сначала проверьте на игрушечном примере, затем примените для 10 и 150 пользователей. 
# 
# Обратите внимание, что в коротких сессиях, меньше 10 сайтов, у нас остались нули, так что первый признак (сколько раз попался 0) по смыслу отличен от остальных (сколько раз попался сайт с индексом $i$). Поэтому первый столбец разреженной матрицы надо будет удалить. 

#%%
X_toy, y_toy = train_data_toy.iloc[:, :-1].values, train_data_toy.iloc[:, -1].values


#%%
X_toy


#%%
X_sparse_toy = csr_matrix ''' ВАШ КОД ЗДЕСЬ '''   

#%% [markdown]
# **Размерность разреженной матрицы должна получиться равной 11, поскольку в игрушечном примере 3 пользователя посетили 11 уникальных сайтов.**

#%%
X_sparse_toy.todense()


#%%
X_10users, y_10users = train_data_10users.iloc[:, :-1].values,                        train_data_10users.iloc[:, -1].values
X_150users, y_150users = train_data_150users.iloc[:, :-1].values,                          train_data_150users.iloc[:, -1].values


#%%
X_sparse_10users = ''' ВАШ КОД ЗДЕСЬ '''
X_sparse_150users = ''' ВАШ КОД ЗДЕСЬ '''

#%% [markdown]
# **Сохраним эти разреженные матрицы с помощью [pickle](https://docs.python.org/2/library/pickle.html) (сериализация в Python), также сохраним вектора *y_10users, y_150users* – целевые значения (id пользователя)  в выборках из 10 и 150 пользователей. То что названия этих матриц начинаются с X и y, намекает на то, что на этих данных мы будем проверять первые модели классификации.
# Наконец, сохраним также и частотные словари сайтов для 3, 10 и 150 пользователей.**

#%%
with open(os.path.join(PATH_TO_DATA, 'X_sparse_10users.pkl'), 'wb') as X10_pkl:
    pickle.dump(X_sparse_10users, X10_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'y_10users.pkl'), 'wb') as y10_pkl:
    pickle.dump(y_10users, y10_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'X_sparse_150users.pkl'), 'wb') as X150_pkl:
    pickle.dump(X_sparse_150users, X150_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'y_150users.pkl'), 'wb') as y150_pkl:
    pickle.dump(y_150users, y150_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'site_freq_3users.pkl'), 'wb') as site_freq_3users_pkl:
    pickle.dump(site_freq_3users, site_freq_3users_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'site_freq_10users.pkl'), 'wb') as site_freq_10users_pkl:
    pickle.dump(site_freq_10users, site_freq_10users_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'site_freq_150users.pkl'), 'wb') as site_freq_150users_pkl:
    pickle.dump(site_freq_150users, site_freq_150users_pkl, protocol=2)

#%% [markdown]
# **Чисто для подстраховки проверим, что число столбцов в разреженных матрицах `X_sparse_10users` и `X_sparse_150users` равно ранее посчитанным числам уникальных сайтов для 10 и 150 пользователей соответственно.**

#%%
assert X_sparse_10users.shape[1] == len(site_freq_10users)


#%%
assert X_sparse_150users.shape[1] == len(site_freq_150users)

#%% [markdown]
# На следующей неделе мы еще немного поготовим данные и потестируем первые гипотезы, связанные с нашими наблюдениями. 

