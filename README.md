# DA_final

# Поиск клиентов с неоптимальным тарифом

## План работы над проектом

**1. Предобработка данных**

- [x] Изучение пропусков данных
- [x] Проверка на явные и скрытые дубликаты
- [x] Объединение таблиц в единый датасет

**2. Исследовательский анализ**

**2.1 Изучить временной период данных:**
- [x] как менялось количество клиентов
- [x] какой средний срок жизни активного клиента 

**2.2 Составить портрет клиента:**
- [x] Количество операторов
- [x] количество активных и неактивных операторов среди активных клиентов
- [x] выделить категорию клиентов на основании категории операторов: обзвоны (преимущественно внешние исходящие звонки, OUT) и служба поддержки (преимущественно внешние входящие звонки, IN), добавить столбец с обозначением категории. 
- [x] Составить сводную таблицу: клиент / тариф/ количество операторов/ плата за 1 оператора/ продожительность звонков OUT/ стоимость тарифа OUT/ количество звонков OUT INTERNAL/ лимит внутренних звонков/ стоимость тарифа OUT INTERNAL/ абонентская плана/ оплата за операторов_total/ оплата за исходящие звонки_total/ оплата за внутренние звонки_total/ финальная сумма за месяц

**2.3 Составить краткую картину тарифных планов:**
- [x] количество клиентов
- [x] среднее количество операторов у 1 клиента
- [x] средний доход в месяц
- [x] Составить текущую финансовую картину: 
- [x] средний доход в месяц с клиента,
- [x] как менялась средняя месячная выручка компании за весь период времени
- [x] Сравним текущие доходы и ожидаемые в случае смены тарифа имеющимся клиентами

**3. Проверка гипотез**
- [x] Уровень дополнительного дохода за счет использования клиентами неоптимального тарифа не отличается для разных тарифных планов
- [x] Средний доход от клиентов колл-центра (преимущественно внешние исходящие звонки) и службы поддержки (преимущественно внешние входящие звонки) не отличаются

**4. Краткие выводы и рекомендации**

**5.Подготовка презентационных материалов**

## Подготовительный этап

# импорт библиотек
import pandas as pd 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

clients = pd.read_csv('https://code.s3.yandex.net/datasets/telecom_clients.csv')
clients

df = pd.read_csv('https://code.s3.yandex.net/datasets/telecom_dataset.csv')
df

df.info()

df.isnull().sum()

df.duplicated().sum()

**Краткий обзор данных**

Датасет содержит данные о действиях клиентов в сервисе интернет-телефонии

- Состоит из 53 902 строк
- В столбцах *internal* и *operator_id* содержатся пропуски
- Выявлено 4 900 полных дубликатов (9%)
- Необходимо перевести данные: 
    - в столбце *date* в формат datetime, 
    - в столбце *internal* в формат bool, 
    - в столбце *operator_id* в формат int64

## Предобработка данных

#заполним пропуски в стобцах на 0 для дальнейшей обработки
df['operator_id'] = df['operator_id']. fillna (0)
df['internal'] = df['internal'].fillna (0)

<div class="alert alert-success">
    
#приведем дату наблюдения (столбец date) в формат datetime

df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

#приведем cтолбец internal в формал bool
df['internal'] = df['internal'].astype('bool')

#приведем cтолбец operator_id в формал int64
df['operator_id'] = df['operator_id'].astype('int64')

df.info()

#проверим столбец direction на неявные дубликаты
df['direction'].unique()

#проверим столбец internal на неявные дубликаты
df['internal'].unique()

#на предыдущем этапе мы обнаружили почти 5 тыс дубликатов. удалим их для дальнейшей работы

df = df.drop_duplicates().reset_index(drop=True)

# проверка
print('Количество дубликатов = ', df.duplicated().sum())

df.isnull().sum()

df_operator_null = df[df['operator_id'] == 0]
df_operator_null

#проверим есть ли зависимость отсутствия id оператора от типа звонка
print(df_operator_null['direction'].value_counts())

print('')

#проверим есть ли зависимость отсутствия id оператора от типа звонка
print(df_operator_null['internal'].value_counts())

Пока не понятно, как можно заполнить пропуски по столбцу operator_id, оставим в текущем виде

#проверим датасет на аномалии по количеству и продолжительности звонков
df_calls = df[['calls_count', 'call_duration', 'total_call_duration']]
df_calls.describe()

#построим диаграмму распределения

df_calls.boxplot()
plt.title('Распределение количества и длительности звонков')
plt.ylabel('Количество')
plt.show()

На графике видно большое количество выбросов, однако стоит отметить тот факт, что звонки длительностью меньше 1 минуты приравниваются к 1 минуте разговора, т.е. если оператор совершил 4 исходящих звонка в минуту, то в таблице это будет записано как 4 минуты. Этим можно объяснить выбросы в столбце количества звонков.  
Столбцы call_duration и total_call_duration агрегированные, но даже в сумме вряд ли можно столько наговорить даже за 24 часа. Удалим записи, где значения в ячейке больше значения 97 перцентиля соответствующего столбца - так мы потеряем не много данных.

#удаляем аномалии в столбцах call_duration и total_call_duration 

df = (df.loc[(df['call_duration'] <= df['call_duration'].quantile(.97)) 
             & (df['total_call_duration'] <= df['total_call_duration'].quantile(.97))])
df

#объединим таблицы в единый датасет
df =  df.merge(clients, on='user_id', how='inner')
df

#приведем столбец с датой регистрации к формату datetime
df['date_start'] = pd.to_datetime(df['date_start'])
df.info()

**Краткий обзор данных**

- Заменили пропуски в столбцах на 0 для дальнейшей обработки
- Привели данные к необходимому типу данных
- Удалили полные дубликаты
- Выявили аномалии и удалили из дальнейшей работы

## Исследовательский анализ данных

### Исследование временного периода и активности клиентов

#изучим временной отрезок данных

print ('Минимальная дата:', df['date'].min())
print ('Максимальная дата:', df['date'].max())
period = df['date'].max() - df['date'].min()
print('Временной период', period)

В нашем распоряжении данные за период с 2 августа по 28 ноября 2019 года (4 месяца)

# данные по дате и вычисляем количество звонков
call_by_date = df.groupby('date')['calls_count'].count().reset_index()
call_by_date

# Создаем столбчатую диаграмму с использованием Seaborn
plt.figure(figsize=(18, 6))
sns.barplot(data = call_by_date, x='date', y='calls_count', palette='rocket_r')
plt.title('Количество звонков по датам', fontsize=16)
plt.xlabel('Дата')
plt.xticks(rotation=90)
plt.ylabel('Количество звонков', fontsize=14)
plt.show()

На графике явно прослеживается тенденция активных звонков в будние дни и минимум звонков в выходные

#добавим столбец с днями недели
df['day_of_week'] = df['date']. dt.weekday
df

# соберем данные по дню недели и вычмислим количество звонков
calls_by_day_of_week = df.groupby('day_of_week')['calls_count'].count().reset_index()
calls_by_day_of_week

# Создаем столбчатую диаграмму с использованием Seaborn
plt.figure(figsize=(18, 4))
sns.barplot(data=calls_by_day_of_week, x='day_of_week', y='calls_count', palette='rocket_r')
plt.title('Количество звонков по дням недели', fontsize=16)
plt.xlabel('Дата')
plt.ylabel('Количество звонков')

plt.show()

# данные по дате и вычисляем количество клиентов
client_by_date = df.groupby('date')['user_id'].nunique().reset_index()
client_by_date

# Создаем столбчатую диаграмму с использованием Seaborn
plt.figure(figsize=(18, 6))
sns.barplot(data=client_by_date, x='date', y='user_id', palette='rocket_r')
plt.title('Количество клиентов по датам', fontsize=16)
plt.xlabel('Дата', fontsize=14)
plt.xticks(rotation=90)
plt.ylabel('Количество клиентов', fontsize=14)
plt.show()

Количество клиентов постепенно увеличивается, при этом мы знаем, что наш заказчик давно на рынке и в начале анализа не может быть такого низкого числа клиентов.  
Посмотрим, что по датам регистрации клиентов

# данные по дате регистрации и вычисляем количество клиентов
client_by_start_date = df.groupby('date_start')['user_id'].nunique()
client_by_start_date

client_by_start_date.describe()

# Создаем столбчатую диаграмму с использованием Seaborn
plt.figure(figsize=(18, 5))
sns.barplot(x=client_by_start_date.index, y=client_by_start_date.values, palette='rocket_r')
plt.title('Количество новых клиентов в день', fontsize=16)
plt.xlabel('Дата')
plt.xticks(rotation=90)
plt.ylabel('Количество новых клиентов')
plt.show()

Выявлено, что в датасете включены только новые клиенты, зарегистрировавшиеся в системе в период с 1 августа по 31 октября 2019 года, поэтому такие низкие показатели в начале периода наблюдения.  
В среднем в день было зарегистрировано 5 новых клиента

#посмотрим среднее количество клиентов в каждом месяце
client_by_date['month'] = pd.to_datetime(client_by_date["date"]).dt.month
client_by_date.groupby('month')['user_id'].mean()

#поскольку в нашем датасете используются данные по клиентам, которые начали регистрироваться в системе с 1 августа,
#то мы наблюдаем в августе крайне низкие цифры по количеству клиентов в день (18 чел) при цифрах в октябре и ноябре свыше 120 чел.

# удалим данные за август из дальнейшего анализа
df['active_month']=pd.to_datetime(df["date"]).dt.month
df=df.loc[df['active_month'] != 8]
df['active_month'].value_counts()

#составим сводную таблицу по датам активности пользователя(от даты регистрации до даты последней активности)
period_of_user_life = df.groupby('user_id').agg({'date':'last', 'date_start':'first'}).reset_index()
period_of_user_life['lifetime'] = period_of_user_life['date']-period_of_user_life['date_start']
period_of_user_life['lifetime'] = period_of_user_life['lifetime'].astype('str')
period_of_user_life['lifetime'] = period_of_user_life['lifetime'].str.replace(' days', '')
period_of_user_life['lifetime'] = period_of_user_life['lifetime'].astype('int64')
period_of_user_life = period_of_user_life.sort_values(by ='lifetime', ascending=False)
period_of_user_life

#построим гистограмму по жизненному циклу клиентов
period_of_user_life['lifetime'].hist(bins=30, figsize=(15, 5))
plt.title('Гистограмма по жизненному циклу клиентов')
plt.ylabel('Количество клиентов')
plt.xlabel('Продолжительность жизненного цикла')
plt.show()

#выделим отлельные столбцы по месяцу регистрации и месяцу последней активности
period_of_user_life["first_month"] = pd.to_datetime(period_of_user_life["date_start"]).dt.month
period_of_user_life["last_month"] = pd.to_datetime(period_of_user_life["date"]).dt.month
period_of_user_life

#построим тепловую карту продолжительности жизни клиентов
period_of_user_life_matrix = period_of_user_life.pivot_table(index='first_month', 
                                                             columns='last_month', 
                                                             values='user_id', 
                                                             aggfunc='nunique', 
                                                             fill_value=0)

sns.heatmap(period_of_user_life_matrix, cmap='rocket_r', annot=True, fmt='d', cbar=False)
plt.title('Жизненный цикл клиентов', fontsize=16)
plt.xlabel('Месяц последней активности', fontsize=14)
plt.ylabel('Месяц регистрации', fontsize=14)
plt.show()

Большинство клиентов сотаются с нами до настоящего момент, однако 41 клиент (13%) не стали продлевать с нами сотрудничество.  
Рассмотрим их подробнее

#выделим неактивных пользователей в ноябре
inactive_user = period_of_user_life.loc[period_of_user_life['last_month'] !=11]
inactive_user.reset_index(drop=False)

41 пользователь из 303 (13%) не оформляли подписку на ноябрь 2019 года, исключим их из дальнейшего анализа.

Однако, отдельно рассмотрим пользователей (23 клиента или 8 %), которые прекратили подписку в октябре и сентябре 2019 года и имеют продолжительность активности более 30 дней - это "потерянные клиенты", возможно именно их мы сможем вернуть, если предложим привлекательные условия

#отдельно посмотрим пользователей с активностью меньше 28 дней
inactive_user_2 = period_of_user_life.loc[(period_of_user_life['last_month'] == 11) & (period_of_user_life['lifetime'] <28)]
inactive_user_2.sort_values(by='date')

В целом таких пользователей немного, преимущественно те, кто присоединился к нашему сервису в конце октября

#выделим в отдельный датафрейм потерянных клиентов
lost_user = inactive_user.loc[inactive_user['lifetime']>=30]
lost_user = lost_user[['user_id','lifetime']]
lost_user = lost_user.merge(df, on='user_id', how='inner') 
lost_user

# вернемся к нему позже

#выделим датафрейм с активными пользователями, с которым продолжим дальнейшую работу

active_user = period_of_user_life.loc[period_of_user_life['last_month'] ==11]
active_user = active_user[['user_id','lifetime']]
active_user = active_user.merge(df, on='user_id', how='inner') 
active_user

#посмотрим на активность клиентов по количеству звонков
active_user_1 = active_user.groupby('user_id').agg({'calls_count': 'sum'}).sort_values(by='calls_count').reset_index()
active_user_1

#добавим в основной датасет информацию о количестве звонков
active_user = active_user.merge(active_user_1, on='user_id', how='inner')

#рассчитаем среднее количество звонков в рабочий день по каждому клиенту
active_user['mean_call_count']=active_user['calls_count_y']/active_user['lifetime']
active_user

active_user['mean_call_count'].describe()

plt.figure(figsize=(20, 6))
sns.boxplot(x='mean_call_count', data=active_user, palette="rocket")
plt.xlabel("Среднее количество звонков в день")
plt.title("Распределение звонков")
plt.show()

На диаграмме размаха видны выбросы в виде крайне высоких значений, однако в нашем случае это могут быть крупные клиенты с большим количеством операторов, поэтому на данном этапе мы удалять их не будем. вернемся к ним позже.
однако сейчас мы удалим неактивных клиентов с нулевым значением

#исключим неактивных клиентов
active_user = active_user.loc[active_user['mean_call_count'] != 0]
active_user

active_user['user_id'].nunique()

active_user.info()

**Вывод**

- В нашем распоряжении данные за период с 2 августа по 28 ноября 2019 года (4 месяца)
- Прослеживается тенденция активных звонков в будние дни и минимум звонков в выходные
- В датасете включены только новые клиенты, зарегистрировавшиеся в системе в период с 1 августа по 31 октября 2019 года, поэтому такие низкие показатели в начале периода наблюдения. В среднем в день было зарегистрировано 4 новых клиента
- 41 пользователей из 303 (13%) не оформляли подписку на ноябрь 2019 года
- после удаления неактивных пользователей в нашем датафрейме осталось 261 клиент (86% )

### Составим портрет активного клиента

#изучим количество операторов у каждого клиента
operators_count = (active_user.groupby('user_id')
                   .agg({'operator_id': 'nunique'})
                   .sort_values(by='operator_id')
                   .rename(columns={'operator_id':'operator_id_count'})
                   .reset_index())
operators_count

operators_count['operator_id_count'].value_counts()

operators_count['operator_id_count'].describe()

plt.figure(figsize=(15,5))
sns.histplot(data=operators_count, x="operator_id_count", bins = 50, palette = 'rocket')
plt.xlabel("Количество операторов")
plt.ylabel("Количество клиентов")
plt.title("Распределение операторов на 1 клиента")
plt.show()

Преимущественно наши клиенты имеют от 2 до 5 операторов, однако встречаются и крупные компании со штатом оператов свыше 20 человек (6 компаний)

#добавим количество операторов в основную таблицу

active_user = active_user.merge(operators_count, on='user_id', how='inner')
active_user

#соберем отдельную таблицу по активности операторов
operator_lifetime = (active_user
                     .pivot_table(index='operator_id',
                                 values='date',
                                 aggfunc=['first', 'last', 'nunique'])
                     .rename(columns={'date':''})
                     )
operator_lifetime.columns = [t[0] if t[0] else t[1] for t in operator_lifetime.columns]
operator_lifetime['lifetime'] = operator_lifetime['last'] - operator_lifetime['first']
operator_lifetime['lifetime'] = operator_lifetime['lifetime'].astype('str')
operator_lifetime['lifetime'] = operator_lifetime['lifetime'].str.replace(' days', '')
operator_lifetime['lifetime'] = operator_lifetime['lifetime'].astype('int')
operator_lifetime = operator_lifetime.sort_values(by ='lifetime', ascending=False)
operator_lifetime

#построитм гистограмму для распределения лайфтайма операторов
plt.figure(figsize=(15,6))
sns.histplot(data=operator_lifetime, x="nunique", bins = 100, palette = 'rocket')
plt.xlabel("Продолжительность жизни")
plt.ylabel("Количество операторов")
plt.title("Распределение операторов по жизненному циклу")
plt.show()

Отметим большое количество операторов, имеющих менее 7 рабочих смен

#добавим информацию по временным показателям в основную таблицу по операторам
operator_info = active_user[['user_id', 'operator_id','date', 'direction', 'internal','is_missed_call', 'calls_count_x', 'call_duration', 'total_call_duration']]
operator_info=operator_info.merge(operator_lifetime, on ='operator_id', how='inner')
operator_info

#удалим операторов без рабочих дней в ноябре
operator_info = operator_info.loc[operator_info['last'] >= '2019-11-01']
operator_info

operator_info['operator_id'].nunique()

#выделим отдельную таблицу по активности операторов
operator_activity = (operator_info
                     .groupby(['user_id','operator_id', 'nunique'])[['calls_count_x', 'call_duration','total_call_duration']]
                     .agg('sum')
                     .sort_values(by='total_call_duration')
                     .reset_index())
operator_activity['total_call_duration_per_work_day'] = operator_activity['total_call_duration']/operator_activity['nunique']
operator_activity

operator_activity.describe()

plt.figure(figsize=(20, 6))
sns.boxplot(x='total_call_duration_per_work_day', data=operator_activity, palette="rocket")
plt.xlabel("Среднее количество звонков в день")
plt.title("Распределение звонков на 1 оператора")
plt.show()

#выделим из таблицы операторов, у которых общее количество часов равно нулю
#исключать операторов с большим количеством звонков нецелесообразно, т.к. есть вероятность, что под одним id работает несколько человек
operator_activity_1 = operator_activity.query('total_call_duration_per_work_day == 0')
operator_activity_1

#проверим количество операторов с id равным 0 и средней продолжительностью смены равной 0
operator_activity.query('operator_id == 0 & total_call_duration_per_work_day == 0')

#исключим 5 операторов из датафрейма
non_active_operator = [914626, 960674, 946454, 955068, 958458]
operator_activity = operator_activity.query('operator_id not in @non_active_operator')
operator_activity

operator_activity['total_call_duration_per_work_day'].describe()

#исключим 5 операторов из основного датафрейма

operator_info = operator_info.query('operator_id not in @non_active_operator')
operator_info

#исключим из датасета с активными клиентами операторов с нулевыми значениями звонков
active_operator = operator_info['operator_id']
active_user = active_user.query('operator_id in @active_operator')
active_user

#добавим столбец с указанием месяца активности
active_user['month'] = pd.to_datetime(active_user["date"]).dt.month
active_user

# составим таблицу по клиентам в разрезе типа звонков
user_call_pivot = active_user.pivot_table(
    index=['user_id','month'],
    columns=['internal','direction'],
    values='total_call_duration',
    aggfunc=['sum']
).reset_index()
user_call_pivot.columns = ['user_id','month','external_in', 'external_out', 'internal_in', 'internal_out']

user_call_pivot.fillna(0, inplace=True)

user_call_pivot

#добавим столбец с типом клиента

user_call_pivot_1 = (user_call_pivot.groupby('user_id')[['external_in', 'external_out', 'internal_out', 'internal_in']]
                     .sum()
                     .reset_index())
user_call_pivot_1.columns = ['user_id','external_in', 'external_out', 'internal_in', 'internal_out']

user_call_pivot_1 ['client_type'] = ''

for i in range(len(user_call_pivot_1)): 
    if ((user_call_pivot_1.loc[i, 'external_in'] > user_call_pivot_1.loc[i, 'external_out']) 
        & (user_call_pivot_1.loc[i, 'external_in'] > user_call_pivot_1.loc[i, 'internal_in'])
        & (user_call_pivot_1.loc[i, 'external_in'] > user_call_pivot_1.loc[i, 'internal_out'])):
        user_call_pivot_1.at[i, 'client_type'] = 'external_in' 
    elif ((user_call_pivot_1.loc[i, 'external_out'] > user_call_pivot_1.loc[i, 'external_in'])
          & (user_call_pivot_1.loc[i, 'external_out'] > user_call_pivot_1.loc[i, 'internal_in'])
          & (user_call_pivot_1.loc[i, 'external_out'] > user_call_pivot_1.loc[i, 'internal_out'])):
        user_call_pivot_1.at[i, 'client_type'] = 'external_out'
    elif  ((user_call_pivot_1.loc[i, 'internal_in'] > user_call_pivot_1.loc[i, 'external_in'])
          & (user_call_pivot_1.loc[i, 'internal_in'] > user_call_pivot_1.loc[i, 'external_out'])
          & (user_call_pivot_1.loc[i, 'internal_in'] > user_call_pivot_1.loc[i, 'internal_out'])):
        user_call_pivot_1.at[i, 'client_type'] = 'internal_in'      
    else:
        user_call_pivot_1.at[i, 'client_type'] = 'internal_out'

user_call_pivot_1 = user_call_pivot_1[['user_id', 'client_type']]
user_call_pivot_1

#составим матрицу по месяцам и типам звонков
call_pivot = (active_user
                     .pivot_table(index = ['month', 'internal','direction'],
                                 values = 'total_call_duration',
                                 aggfunc = 'sum')
                     .reset_index())

call_pivot['call_type'] = ''

for i in range(len(call_pivot)): 
    if ((call_pivot.loc[i, 'internal'] == False) & (call_pivot.loc[i, 'direction'] == 'in')): 
        call_pivot.at[i, 'call_type'] = 'external_in' 
    elif ((call_pivot.loc[i, 'internal'] == False) & (call_pivot.loc[i, 'direction'] == 'out')):
        call_pivot.at[i, 'call_type'] = 'external_out'
    elif ((call_pivot.loc[i, 'internal'] == True) & (call_pivot.loc[i, 'direction'] == 'in')):
        call_pivot.at[i, 'call_type'] = 'internal_in'      
    else:
        call_pivot.at[i, 'call_type'] = 'internal_out'
          
call_pivot=call_pivot[['month', 'call_type', 'total_call_duration']]
call_pivot

plt.figure(figsize=(18, 10))
ax = sns.barplot(x='month', y='total_call_duration', hue='call_type', data=call_pivot, palette = 'rocket')
plt.title('Продолжительность звонков по месяцам и типу')
plt.xlabel('Продолжительность звонков')
plt.ylabel('Месяц')
plt.xticks(rotation=0)
plt.show()

Количество внешних исходящих звонков значительно превышает все остальные категории в каждом месяце (от 59% в октябре до 63% в сентябре и ноябре)

Количество внешних входящих звонков составляет порядка 35 % ежемесячно (33,8% в сентябре, 38,6% в октябре, 33,5% в ноябре)

Внутренние звонки не пользуются большой популярностью, преимущественно это внутренние исходящие звонки (2,5% ежемесячно)


user_call_pivot.describe()

В описании значений по внутренним входящим звонкам 75% записей равны 0, при этом максимум зафиксирован на значении 29 507.
Схожая ситуация наблюдается среди внутренних исходящих: 3 квартиль отмечен на значении 2, при этом максимум зафиксирован на значении 83 569.  

Рассмотрим отдельно максимальные значения по этим типам звонков

user_call_pivot_int_in = user_call_pivot.query('internal_in > 1000')
user_call_pivot_int_in

Максимальные значения зафиксированы у наиболее крупных клиентов, все ок

user_call_pivot_int_out = user_call_pivot.query('internal_out > 1000')
user_call_pivot_int_out

Интересный момент, исходящих внутренних звонков гораздо больше входящих.
Рабочая гипотеза, что это перераспределение звонков или звонки менеджеру. Для проверки этого предположения необходимо запросить дополнительные данные

#сделаем срез по количеству операторов среди клиентов разных тарифных планов
operators_per_client = active_user[['user_id','operator_id_count', 'tariff_plan']]
operators_per_client = operators_per_client.drop_duplicates().reset_index(drop=True)
operators_per_client

#для дальнейшей работы с тарифными планами сведем в одну таблицу информацию по клиентам, операторам с указанием тарифных планов
user_call_pivot = user_call_pivot.merge(user_call_pivot_1, on='user_id', how='left') 
user_call_pivot = user_call_pivot.merge(operators_per_client, on='user_id', how='left')
user_call_pivot

#проверим количество клиентов разных типов на каждом тарифе
tariff_by_client_type = (user_call_pivot
                         .pivot_table(index = ['tariff_plan','client_type'],
                                      values = 'user_id',
                                      aggfunc = 'nunique')
                     .reset_index())

tariff_by_client_type

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='tariff_plan', y='user_id', hue='client_type', data=tariff_by_client_type, palette = 'rocket')
plt.title('Количество клиентов по типу клиентов')
plt.xlabel('Тарифный план')
plt.ylabel('Количество клиентов')
plt.show()

На тарифном плане А преобладают клиенты категории "колл-центр" (преимущественно внешние исходящие звонки)

Тариф план В и С характеризуется большим количеством клиентов "службы поддержки" (преимущестенно внешние входящие звонки)

**Краткие выводы:**

- Преимущественно наши клиенты имеют от 1 до 7 операторов, однако встречаются и крупные компании со штатом оператов свыше 20 человек (6 компаний)
- 783 оператора (83%) работали в ноябре,  среди них выявлено и удалено 7 аномальных id

### Составим картину тарифных планов

#определим количество клиентов на каждом тарифном плане
client_by_tariff = (user_call_pivot
                    .pivot_table(index = ['tariff_plan', 'month'],
                                 values = 'user_id',
                                 aggfunc = 'nunique')
                    .reset_index())
client_by_tariff

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='user_id', hue = 'tariff_plan', data=client_by_tariff, palette = 'rocket')
plt.title('Количество клиентов на каждом  тарифе')
plt.xlabel('Месяц')
plt.ylabel('Количество клиентов')
plt.show()

Самый популярный ***тариф С***: на нем зарегистрировано 135 клиентов (52%),  
на ***тарифе В*** - 96 клиентов (37%)  
и ***тариф А*** с наименьшим количеством клиентов - 30 человек (11%)

#опеределим медианное значение количества операторов у одного клиента
user_call_pivot_short = user_call_pivot[['user_id','operator_id_count','tariff_plan']]
user_call_pivot_short= user_call_pivot_short.drop_duplicates().reset_index(drop=True)
operator_by_tariff = (user_call_pivot_short
                    .pivot_table(index = ['tariff_plan'],
                                 values = 'operator_id_count',
                                 aggfunc = 'median')
                    .reset_index())
operator_by_tariff

Самые крупные клиенты подключают тариф А, среди них медианное значение количества операторов более 4 человек (при этом ранее мы определили, что в среднем у наших клиентов в штате до 3 операторов)

#переведем информацию по тарифным планам в цифровой формат
tariffs = pd.DataFrame({'tariff_plan': ['A', 'B', 'C'], 
                      'internal_limit_min': [2000, 500 ,0],
                     'month_cost': [4000, 2000, 1000], 
                     'internal_cost': [0.1, 0.15, 0.3], 
                     'not_internal_cost': [0.3, 0.5, 0.7], 
                     'operator_cost': [50, 150, 300]})
tariffs

#добавим составляющие тарифного плана в основную таблицу с клиентами
user_call_pivot = user_call_pivot.merge(tariffs, on='tariff_plan', how='left')
user_call_pivot

#добавим столбец с превышением лимита минут по внутренним звонкам
user_call_pivot['over_internal_limit'] = user_call_pivot['internal_out']-user_call_pivot['internal_limit_min']

#заменим отрицательные значения в новом столбце на 0, поскольку эти клиенты уложились в лимит, а неиспользованные минуты нас не интересуют
for col in user_call_pivot.columns:
    user_call_pivot['over_internal_limit'] = np.where(user_call_pivot['over_internal_limit']>0, user_call_pivot['over_internal_limit'], 0)
    
user_call_pivot

#рассчитаем итоговую сумму платежа за каждый месяц по клиенту
user_call_pivot['revenue'] = (user_call_pivot['month_cost'] +
                              user_call_pivot['over_internal_limit'] * user_call_pivot['internal_cost'] + 
                              user_call_pivot['external_out'] * user_call_pivot['not_internal_cost'] +
                              user_call_pivot['operator_id_count'] * user_call_pivot['operator_cost'])
user_call_pivot

#помесячная выручка с каждого тарифа
month_revenue = (user_call_pivot
                 .pivot_table (index = ['month','tariff_plan'],
                              values = 'revenue',
                              aggfunc = sum)
                 .reset_index())
month_revenue

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='revenue', hue='tariff_plan', data=month_revenue, palette = 'rocket')
plt.title('Ежемесячная выручка по каждому тарифу')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.show()

Наибольшую выручку нам приносят клиенты тарифа С, однако и количество клиентов на этом тарифе гораздо выше А и В.

Посмотрим, среднюю выручку на 1 клиента на каждом тарифном плане

#средняя выручка в месяц на 1 клиента тарифного плана
month_revenue_per_client = (user_call_pivot
                 .pivot_table (index = ['month','tariff_plan'],
                              values = ['user_id', 'revenue'],
                              aggfunc = {'user_id':'nunique', 'revenue': 'sum'})
                 .reset_index())
month_revenue_per_client['mean_revenue'] = month_revenue_per_client['revenue']/month_revenue_per_client['user_id']
month_revenue_per_client

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='mean_revenue', hue='tariff_plan', data=month_revenue_per_client, palette = 'rocket')
plt.title('Средняя выручка на 1 клиента')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.show()

Выручка от клиентов тарифа А в течение 3 месяцев стремительно росла (почти в 8 раз).

Выручка от клиентов тарифов В и С так же показывает рост, однако горазно более плавный:  
тариф В - с 12,2 тыс до 17,3 тыс (+40%)  
тариф С - 12,0 тыс до 15,0 тыс (+25%)

# посмотрим по каким составляющим тарифного плана происходит рост выручки
tariff_component = user_call_pivot[['user_id','month', 'external_in', 'external_out', 'internal_in', 'internal_out', 
                                    'client_type', 'operator_id_count', 'tariff_plan', 'internal_limit_min',
                                   'internal_cost', 'not_internal_cost', 'operator_cost', 'over_internal_limit']]

tariff_component['revenue_over_internal_limit'] =  tariff_component['over_internal_limit'] * tariff_component['internal_cost']
tariff_component['revenue_external_out'] = tariff_component['external_out'] * tariff_component['not_internal_cost']
tariff_component['revenue_operator'] = tariff_component['operator_id_count'] * tariff_component['operator_cost']

tariff_component = tariff_component[['user_id','month','client_type', 'tariff_plan', 'revenue_over_internal_limit',
                                    'revenue_external_out', 'revenue_operator']]

tariff_component

month_revenue_per_components = (tariff_component
                                .groupby(['month','tariff_plan'])
                                ['user_id','revenue_over_internal_limit', 'revenue_external_out', 'revenue_operator']
                                .agg({'user_id': 'nunique', 
                                      'revenue_over_internal_limit':'sum',
                                      'revenue_external_out':'sum',
                                      'revenue_operator':'sum'})
                                .reset_index())

month_revenue_per_components['revenue_over_internal_limit_per_client'] = (month_revenue_per_components['revenue_over_internal_limit']
                                                                    /month_revenue_per_components['user_id'])
month_revenue_per_components['revenue_external_out_per_client'] = (month_revenue_per_components['revenue_external_out']
                                                                    /month_revenue_per_components['user_id'])
month_revenue_per_components['revenue_operator_per_client']= (month_revenue_per_components['revenue_operator']
                                                                    /month_revenue_per_components['user_id'])
month_revenue_per_components

plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='revenue_over_internal_limit_per_client', hue='tariff_plan', data=month_revenue_per_components, palette = 'rocket')
plt.title('Выручка по превышению лимита внутренних звонков')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.show()

plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='revenue_external_out_per_client', hue='tariff_plan', data=month_revenue_per_components, palette = 'rocket')
plt.title('Выручка по количеству минут внешних исходящих звонков')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.show()

plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='revenue_operator_per_client', hue='tariff_plan', data=month_revenue_per_components, palette = 'rocket')
plt.title('Выручка по количеству операторов')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.show()

У клиентов тарифа А в ноябре 2019 г наблюдается резкий рост переплаты за превышение литита внутренних звонков (в 13 раз) и выручки за исходящие внешние звонки (в 37 раз) при этом среди клиентов других тарифов таких всплском не наблюдается.
Вероятно, такая картина связана с некорректным выбором тарифа для крупных клиентов

#рассчитаем для каждого клиента затраты на каждом тарифе
user_call_pivot['A_hypo_revenue'] = (4000 + 
                                     user_call_pivot['over_internal_limit'] * 0.1 + 
                                     user_call_pivot['external_out'] * 0.3 +
                                     user_call_pivot['operator_id_count'] * 50)
user_call_pivot['B_hypo_revenue'] = (2000+
                                     user_call_pivot['over_internal_limit'] * 0.15 + 
                                     user_call_pivot['external_out'] * 0.5 +
                                     user_call_pivot['operator_id_count'] * 150)
user_call_pivot['C_hypo_revenue'] = (1000+
                                     user_call_pivot['over_internal_limit'] * 0.3 + 
                                     user_call_pivot['external_out'] * 0.7 +
                                     user_call_pivot['operator_id_count'] * 300)
user_call_pivot

#сделаем выжимку из основной таблицы
df_tariff = user_call_pivot[['user_id','month', 'tariff_plan','revenue','A_hypo_revenue', 'B_hypo_revenue', 'C_hypo_revenue']]
df_tariff

#добавим столбец с оптимальным тарифом для каждого клиента
df_tariff['optimal_tariff'] = '' 
for i in range(len(df_tariff)): 
    if ((df_tariff.loc[i, 'A_hypo_revenue'] < df_tariff.loc[i,'B_hypo_revenue']) & (df_tariff.loc[i, 'A_hypo_revenue'] < df_tariff.loc[i,'C_hypo_revenue'])): 
        df_tariff.at[i, 'optimal_tariff'] = 'A' 
    elif ((df_tariff.loc[i, 'B_hypo_revenue'] < df_tariff.loc[i,'A_hypo_revenue']) & (df_tariff.loc[i, 'B_hypo_revenue'] < df_tariff.loc[i,'C_hypo_revenue'])):
        df_tariff.at[i, 'optimal_tariff'] = 'B'
    else:
        df_tariff.at[i, 'optimal_tariff'] = 'C' 
        
df_tariff

#проверим на наличие разных оптимальных тарифов в разных месяцах
double_tariff = df_tariff.groupby('user_id')['optimal_tariff'].nunique().reset_index()
double_tariff=double_tariff.query('optimal_tariff > 1')
len(double_tariff)

#добавим критерий "оптимальный-неоптимальный тариф"
def custom_func(row):
    if row['optimal_tariff'] != row['tariff_plan']:
        return 'Выявлен неоптимальный тариф'
    else:
        return 'OK'

df_tariff['allarm'] = df_tariff.apply(custom_func, axis=1)
df_tariff

non_optimal_tariff = (df_tariff.pivot_table(index = ['month', 'allarm'],
                               values = 'user_id',
                               aggfunc = 'nunique')
                       .reset_index())
non_optimal_tariff

#посмотрим на диаграмме распределение неоптимального тарифа
plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='user_id', hue = 'allarm', data=non_optimal_tariff, palette = 'rocket')
plt.title('Соотношение оптимального и неоптимального тарифа')
plt.xlabel('Месяц')
plt.ylabel('Количество клиентов')
plt.show()

non_optimal_tariff_1 = (df_tariff.pivot_table(index = 'month',
                               columns = 'allarm',
                               values = 'user_id',
                               aggfunc = 'nunique')
                       .reset_index())
non_optimal_tariff_1.columns = ['month', 'ok', 'non_optimal']
non_optimal_tariff_1['non_optimal_part'] = non_optimal_tariff_1['non_optimal']/ (non_optimal_tariff_1['ok'] + non_optimal_tariff_1['non_optimal'])
non_optimal_tariff_1

#посмотрим на диаграмме распределение неоптимального тарифа
plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='non_optimal_part', data=non_optimal_tariff_1, palette = 'rocket')
plt.title('Доля неоптимального тарифа')
plt.xlabel('Месяц')
plt.ylabel('Доля')
plt.show()

Несмотря на рост в абсолютных значениях, доля клиентов с неоптимальным тарифом ежемесячно остается в диапазоне от 50 до 60 %

#рассмотрим клиентов с оптимальными тарифами в разрезе каждого тарифного плана
optimal_tariff_info = (df_tariff.pivot_table(index = ['month','tariff_plan'],
                                              columns='optimal_tariff',
                                              values = 'allarm',
                                              aggfunc = 'count')
                       .reset_index())
optimal_tariff_info.columns = ['month', 'tariff_plan', 'A','B','C']
optimal_tariff_info['all'] = optimal_tariff_info['A'] + optimal_tariff_info['B'] + optimal_tariff_info['C']

#добавим столбец с долей клиентов с оптимальным тарифом

optimal_tariff_info['optimal_part'] = '' 

for i in range(len(optimal_tariff_info)): 
    if (optimal_tariff_info.loc[i, 'tariff_plan'] == 'A'): 
        optimal_tariff_info.at[i, 'optimal_part'] = (optimal_tariff_info.loc[i, 'A'] / optimal_tariff_info.loc[i, 'all'])                                            
    elif (optimal_tariff_info.loc[i, 'tariff_plan'] == 'B'):
        optimal_tariff_info.at[i, 'optimal_part'] =  (optimal_tariff_info.loc[i, 'B'] / optimal_tariff_info.loc[i, 'all'])
    else:
        optimal_tariff_info.at[i, 'optimal_part'] = (optimal_tariff_info.loc[i, 'C'] / optimal_tariff_info.loc[i, 'all'])

optimal_tariff_info

#посмотрим на диаграмме распределение оптимального тарифа
plt.figure(figsize=(18, 7))
ax = sns.barplot(x='month', y='optimal_part', hue = 'tariff_plan', data=optimal_tariff_info, palette = 'rocket')
plt.title('Динамика доли клиентов с оптимальным тарифом')
plt.xlabel('Месяц')
plt.ylabel('Доля клиентов')
plt.show()

Доля клиентов тарифного плана А, для которых использование данного тарифа оптимально, качественно возрастает от месяца к месяцу - с 25% в сентябре до 63 % в ноябре

Доля клиентов тарифного плана В, для которых использование данного тарифа оптимально, напротив снижается - от 21 % в сентябре до 11 % в ноябре. 

Доля клиентов тарифногог плана С, для которых использование данного тарифа оптимально, стабильно держится в районе 60 %.

Необходимо уделить особое внимание тарифному плану В: он не подходит 90% использующих его клиентов. 

tariff_plan_now = (df_tariff.pivot_table(index = ['month','tariff_plan'],
                                           values = 'user_id',
                                           aggfunc = 'nunique')
                       .reset_index())
tariff_plan_now

#посмотрим на диаграмме текущее распределение клиентов по тарифам
plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='user_id', hue = 'tariff_plan', data=tariff_plan_now, palette = 'rocket')
plt.title('Текущее распределение клиентов по тарифным планам')
plt.xlabel('Месяц')
plt.ylabel('Количество клиентов')
plt.show()

tariff_plan_new = (df_tariff.pivot_table(index = ['month','optimal_tariff'],
                                           values = 'user_id',
                                           aggfunc = 'nunique')
                       .reset_index())
tariff_plan_new

#посмотрим на диаграмме распределение клиентов при переходе на оптимальный тариф

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='user_id', hue = 'optimal_tariff', data=tariff_plan_new, palette = 'rocket')
plt.title('Рапределение клиентов при переходе на оптимальный тариф')
plt.xlabel('Месяц')
plt.ylabel('Количество клиентов') 
plt.show()

При сопоставлении текущей и гипотетической картины распределения клиентов по тарифным планам наблюдаем резкое увеличесние клиентов на тарифе А, и значительное снижение клиентов тарифа В.  
Возвращаемся к гипотезе, что с тарифом В необходимо поработать более детально

#дополним нашу тарифную таблицу информацией об оптимальном тарифе

def new_price(row):
    if row['optimal_tariff'] == 'A':
        return row['A_hypo_revenue']
    elif row['optimal_tariff'] == 'B':
        return row['B_hypo_revenue']
    else:
        return row['C_hypo_revenue']

df_tariff['new_revenue'] = df_tariff.apply(new_price, axis=1)
df_tariff['overpayment'] = df_tariff['revenue'] - df_tariff['new_revenue']
df_tariff

df_tariff_compare = df_tariff.groupby(['month'])[['revenue', 'new_revenue', 'overpayment']].sum().reset_index()
df_tariff_compare['percent'] = df_tariff_compare['new_revenue']/df_tariff_compare['revenue'] * 100-100
df_tariff_compare

dtc = pd.DataFrame({'month': [9,9,10,10,11,11],
    'type':['revenue', 'new_revenue', 'revenue', 'new_revenue', 'revenue', 'new_revenue'],
    'count': [1592069.85, 1032187.80, 3339607.60, 2196580.75, 4945639.85, 3497764.10]})

plt.figure(figsize=(18, 5))
ax = sns.barplot(x='month', y='count', hue = 'type', data=dtc, palette = 'rocket')
plt.title('Изменение доходов')
plt.xlabel('Месяц')
plt.ylabel('Выручка') 
plt.show()

При переводе клиентов на оптимальных для них тариф наша компания могла потерять от 35 % выручки в сентябре до 29 % выручки в ноябре

#посмотрим изменения на тарифных планах

#текущая ситуация с выручкой по тарифам
df_tariff_compare_plan = df_tariff.groupby(['month', 'tariff_plan'])[['revenue']].sum().reset_index()

#выручка после перехода на оптимальный тариф
df_tariff_compare_plan_1 = df_tariff.groupby(['month', 'optimal_tariff'])[['new_revenue']].sum().reset_index()
df_tariff_compare_plan_1.columns = ['month', 'tariff_plan', 'new_revenue']

#сводная таблица
df_tariff_compare_plan = df_tariff_compare_plan.merge(df_tariff_compare_plan_1, on=['month', 'tariff_plan'], how = 'left')
df_tariff_compare_plan['percent'] = df_tariff_compare_plan['new_revenue']/df_tariff_compare_plan['revenue']*100-100
df_tariff_compare_plan

dtc_A = pd.DataFrame({'month': [9,9,10,10,11,11],
    'type':['revenue', 'new_revenue', 'revenue', 'new_revenue', 'revenue', 'new_revenue'],
    'count': [86310.30, 759253.60, 417160.60, 1811037.10, 1266425.10, 3098543.40 ]})

plt.figure(figsize=(18, 4))
ax = sns.barplot(x='month', y='count', hue = 'type', data=dtc_A, palette = 'rocket')
plt.title('Изменение доходов по тарифу А')
plt.xlabel('Месяц')
plt.ylabel('Выручка') 
plt.show()

При переводе клиентов на оптимальный тариф, доходы на тарифе А "взлетели бы в космос": выручка бы увеличилась в 9 раз в сентябре, в 4 раза в октябре и в 2,5 раза в ноябре

dtc_B = pd.DataFrame({'month': [9,9,10,10,11,11],
    'type':['revenue', 'new_revenue', 'revenue', 'new_revenue', 'revenue', 'new_revenue'],
    'count': [687222.55, 112167.30, 1204723.60, 144532.55, 1656081.35, 131257.40]})

plt.figure(figsize=(18, 7))
ax = sns.barplot(x='month', y='count', hue = 'type', data=dtc_B, palette = 'rocket')
plt.title('Изменение доходов по тарифу В')
plt.xlabel('Месяц')
plt.ylabel('Выручка') 
plt.show()

Тариф В, как наимее подходящий кому-либо, показал бы падение доходов до 92%

dtc_С = pd.DataFrame({'month': [9,9,10,10,11,11],
    'type':['revenue', 'new_revenue', 'revenue', 'new_revenue', 'revenue', 'new_revenue'],
    'count': [818537.00, 160766.90, 1717723.40, 241011.10, 2023133.40, 267963.30]})

plt.figure(figsize=(18, 7))
ax = sns.barplot(x='month', y='count', hue = 'type', data=dtc_С, palette = 'rocket')
plt.title('Изменение доходов по тарифу С')
plt.xlabel('Месяц')
plt.ylabel('Выручка') 
plt.show()

Тариф С также сильно упал бы в своей доходности - до 85%

#посмотрим переплаты клиентов

client_overpayment = df_tariff.groupby(['user_id', 'month'])['revenue', 'overpayment'].sum().reset_index()
client_overpayment = client_overpayment.query('overpayment != 0')
client_overpayment

client_overpayment.describe()

#построитм гистограмму для распределения перепалат клиентов
plt.figure(figsize=(15,6))
sns.histplot(data=client_overpayment, x="overpayment", bins = 50, palette = 'rocket')
plt.xlabel("Сумма переплаты")
plt.ylabel("Количество клиентов")
plt.title("Распределение суммы переплат")

plt.show()

#сузим диапазон переплаты
plt.figure(figsize=(15,6))
sns.histplot(data=client_overpayment, x="overpayment", bins = 50,  palette = 'rocket')
plt.xlabel("Сумма переплаты")
plt.ylabel("Количество клиентов")
plt.title("Распределение суммы переплат")
plt.xlim(1, 10000)
plt.show()

По гистограмме и описанию видим, что в 50% случаев переплат не превышает 2000. Посмотрим более детально первые 50%

client_overpayment_first = client_overpayment.query('overpayment < 2000')
client_overpayment_first['percent'] = client_overpayment_first['overpayment']/client_overpayment_first['revenue']*100
client_overpayment_first

client_overpayment_first['percent'].describe()

#вернемся к основному датасету. добавим столбец с процентами
df_tariff['percent'] = df_tariff['overpayment']/df_tariff['revenue']*100

#добавим столбец с новым тарифом. Для тех клиентов, переплата которых меньше 2000 и при этом эта сумма меньше 30% от платежа, 
#оставим их текущий тариф

df_tariff ['new_tariff'] = ''

for i in range(len(df_tariff)): 
    if ((df_tariff.loc[i, 'overpayment'] < 2000) & (df_tariff.loc[i, 'percent'] <= 30)): 
        df_tariff.at[i, 'new_tariff'] = df_tariff.at[i, 'tariff_plan'] 
    else:
        df_tariff.at[i, 'new_tariff'] = df_tariff.at[i, 'optimal_tariff']
        
#добавим столбец с новыми доходами

def newest_price(row):
    if row['new_tariff'] == 'A':
        return row['A_hypo_revenue']
    elif row['new_tariff'] == 'B':
        return row['B_hypo_revenue']
    else:
        return row['C_hypo_revenue']

df_tariff['newest_revenue'] = df_tariff.apply(newest_price, axis=1)
df_tariff

df_tariff_compare_new = df_tariff.groupby(['month'])[['revenue', 'newest_revenue']].sum().reset_index()
df_tariff_compare_new['percent'] = df_tariff_compare_new['newest_revenue']/df_tariff_compare_new['revenue'] * 100-100
df_tariff_compare_new

Что при переводе всех пользователей, что при переводе только коиентов с большой переплатой, значительно падение доходов не изменяется

**Краткие выводы**

- 30 клиентов используют тарифный планы А (11%), 96 клиентов - тарифный план Б (37%) и 135 клиентов - тарифный план С (52%)
- Среднее количество операторов среди клиентов А - 5 человек, среди клиентов Б - 4 человека, среди клиентов С - 3 человека
- У 151 клиента (58%) выявлен неоптимальный тариф
- Среди клиентов тарифа А можем порекомендовать 2 клиентам (6%) перейти на тариф Б, 9 клиентам (30%) перейти на тариф С, 19 клиентов (63%) работают с оптимальным для них тарифом
- Среди клиентов тарифа Б наблюдается самый низкий процент соответствия клиентских потребностей и тарифных платежей: лишь 12 клиентов (12%) работают на оптимальном для низ тарифе, остальным выгоднее было бы перейти на тарифный план А (37 клиентов или 39%) или С (47 клиентов или 49%)
- Среди клиентов тарифа С 79 клиентов (59 %) работают на оптимальном для себя тарифном плане, однако 41 клиенту (31%) можно порекомендовать перейти на тарифный план А , а 15 клиентам (11%) перейти на тарифный план Б
- В целом при переходе клиентов на наиболее выгодные для них тарифные планы наша компания потеряет в месяц до 40 % чистой прибыли

## Проверка гипотез

### Гипотеза № 1

**Уровень дополнительного дохода за счет использования клиентами неоптимального тарифа не отличается для разных тарифных планов**

**Нулевая гипотеза (H0)** - уровень дополнительного дохода за счет использования клиентами неоптимального тарифа статистически значимо не отличается между разными тарифами  

**Альтернативная гипотеза (H1)** - уровень дополнительного дохода за счет использования клиентами неоптимального тарифа статистически значимо отличается между разными тарифами

**Статистическая значимость** - 5%

def tariff_to_another_mannwhit(database, tariff1, tariff2):
    alpha = 0.05  # критический уровень статистической значимости
    results = (stats.mannwhitneyu(df_tariff[df_tariff['tariff_plan'] == tariff1]['overpayment'],
                                 df_tariff[df_tariff['tariff_plan'] == tariff2]['overpayment'])[1])
    print ('Сравнение расходов тарифов {} и {}:'.format(tariff1, tariff2))
    print("p-value: {0:.5f}".format(results))
    
    if results < alpha:
        print('Отвергаем нулевую гипотезу о равенстве средних рейтингов')
    else:
        print('Не отвергаем нулевую гипотезу о равенстве средних рейтингов')
    

tariff_to_another_mannwhit(df_tariff, 'A', 'B')

tariff_to_another_mannwhit(df_tariff, 'A', 'C')

tariff_to_another_mannwhit(df_tariff, 'B', 'C')

**Вывод**:

Для клиентов тарифа В в сопоставлении с тарифами А и С p-value очень маленькое, меньше любого осмысленного alpha, значит, нулевая гипотеза H0 отвергнута: уровень дополнительного дохода с клиентов тарифа В за счет использования неоптимального тарифа статистически значимо отличается от тарифа В и С

При это между тарифами А и С p-value выше статистически значимого значения 5%, значит между этими тарифами уровень дополнительного дохода за счет использования клиентами неоптимального тарифа статистически значимо не отличается 

### Гипотеза № 2

**Средний доход от клиентов колл-центра (преимущественно внешние исходящие звонки) и службы поддержки (преимущественно внешние входящие звонки) не отличаются**

**Нулевая гипотеза (H0)** - уровень дохода от клиентов "колл-центра" и "службы поддержки" статистически значимо не отличается 

**Альтернативная гипотеза (H1)** - уровень дохода от клиентов "колл-центра" и "службы поддержки" статистически значимо отличается 

**Статистическая значимость** - 5%

#добавим в таблицу с тарифами данные по типу клиента
stat_pivot = df_tariff.merge(user_call_pivot_1, on='user_id', how = 'left')
stat_pivot = stat_pivot[['user_id', 'month', 'revenue', 'client_type']]
stat_pivot_ex_in = stat_pivot.query('client_type == "external_in"')
stat_pivot_ex_out = stat_pivot.query('client_type == "external_out"')
stat_pivot_ex_in = stat_pivot_ex_in.groupby('user_id')['revenue'].mean()
stat_pivot_ex_out = stat_pivot_ex_out.groupby('user_id')['revenue'].mean()

alpha = 0.05 

#проведем t-тест Стьюдента для проверки гипотезы
results = stats.ttest_ind(stat_pivot_ex_in, stat_pivot_ex_out)

print(results.pvalue)

#oценка результатов t-теста
if results.pvalue < alpha:
    print('Отвергаем нулевую гипотезу о равенстве доходов медду разными типами клиентов')
else:
    print('Не отвергаем нулевую гипотезу о равенстве доходов медду разными типами клиентов')

## Выводы

- Проанализированы данные за период с 2 августа по 28 ноября 2019 года (4 месяца)

**Клиенты**

- В датасете включены только новые клиенты, зарегистрировавшиеся в системе в период с 1 августа по 31 октября 2019 года, поэтому такие низкие показатели в начале периода наблюдения. В среднем в день было зарегистрировано 4 новых клиента
- 41 пользователей из 303 (13%) не оформляли подписку на ноябрь 2019 года
- После удаления неактивных пользователей в нашем датафрейме осталось 261 клиент (86% )
- Преимущественно наши клиенты имеют от 2 до 5 операторов, однако встречаются и крупные компании со штатом оператов свыше 20 человек (6 компаний)

**Звонки**

- Прослеживается тенденция активных звонков в будние дни и минимум звонков в выходные
- Количество внешних исходящих звонков значительно превышает все остальные категории в каждом месяце (от 59% в октябре до 63% в сентябре и ноябре)
- Количество внешних входящих звонков составляет порядка 35 % ежемесячно (33,8% в сентябре, 38,6% в октябре, 33,5% в ноябре)
- Внутренние звонки не пользуются большой популярностью, преимущественно это внутренние исходящие звонки (2,5% ежемесячно)

**Тарифный планы**

- Самый популярный ***тариф С*** -  на нем зарегистрировано 135 клиентов (52%), на ***тарифе В*** - 96 клиентов (37%) и ***тариф А*** с наименьшим количеством клиентов - 30 человек (11%)
- На тарифном плане А преобладают клиенты категории "колл-центр" (преимущественно внешние исходящие звонки)
- Тариф план В и С характеризуется большим количеством клиентов "службы поддержки" (преимущестенно внешние входящие звонки)
- Наибольшую выручку нам приносят клиенты тарифа С, однако и количество клиентов на этом тарифе гораздо выше А и В.
- Наибольшую выручка на 1 клиента приносит тариф А, более того выручка от клиентов тарифа А в течение 3 месяцев стремительно росла (почти в 8 раз).Выручка от клиентов тарифов В и С так же показывает рост, однако горазно более плавный: тариф В - с 12,2 тыс до 17,3 тыс (+40%), тариф С - 12,0 тыс до 15,0 тыс (+25%)
- У клиентов тарифа А в ноябре 2019 г наблюдается резкий рост переплаты за превышение литита внутренних звонков (в 13 раз) и выручки за исходящие внешние звонки (в 37 раз) при этом среди клиентов других тарифов таких всплском не наблюдается. Вероятно, такая картина связана с некорректным выбором тарифа для крупных клиентов
- Несмотря на рост в абсолютных значениях, доля клиентов с неоптимальным тарифом ежемесячно остается в диапазоне от 50 до 60 %
- Доля клиентов тарифного плана А, для которых использование данного тарифа оптимально, качественно возрастает от месяца к месяцу - с 25% в сентябре до 63 % в ноябре
- Доля клиентов тарифного плана В, для которых использование данного тарифа оптимально, напротив снижается - от 21 % в сентябре до 11 % в ноябре.
- Доля клиентов тарифногог плана С, для которых использование данного тарифа оптимально, стабильно держится в районе 60 %.

**Перевод клиентов на оптимальный тариф**

- При сопоставлении текущей и гипотетической картины распределения клиентов по тарифным планам наблюдаем резкое увеличесние клиентов на тарифе А, и значительное снижение клиентов тарифа В.
- При переводе клиентов на оптимальных для них тариф наша компания могла потерять от 35 % выручки в сентябре до 29 % выручки в ноябре
- При переводе клиентов на оптимальный тариф, доходы на тарифе А "взлетели бы в космос": выручка бы увеличилась в 9 раз в сентябре, в 4 раза в октябре и в 2,5 раза в ноябре
- Тариф В, как наимее подходящий кому-либо, показал бы падение доходов до 92%
- Тариф С также сильно упал бы в своей доходности - до 85%

**Проверка гипотез**

- Проверена гипотеза ***Уровень дополнительного дохода за счет использования клиентами неоптимального тарифа не отличается для разных тарифных планов***. Для клиентов тарифа В в сопоставлении с тарифами А и С p-value очень маленькое, меньше любого осмысленного alpha, значит, нулевая гипотеза H0 отвергнута: уровень дополнительного дохода с клиентов тарифа В за счет использования неоптимального тарифа статистически значимо отличается от тарифа В и С. При это между тарифами А и С p-value выше статистически значимого значения 5%, значит между этими тарифами уровень дополнительного дохода за счет использования клиентами неоптимального тарифа статистически значимо не отличается 

- Проверена гипотеза ***Средний доход от клиентов колл-центра (преимущественно внешние исходящие звонки) и службы поддержки (преимущественно внешние входящие звонки) не отличаются***. Выявлено, что уровень дохода от клиентов "колл-центра" и "службы поддержки" статистически значимо отличается 

**Рекомендации**

- Необходимо уделить особое внимание тарифному плану В: он не подходит 90% использующих его клиентов.
- Провести А/В тестирование на удержание клиентов после перевода на оптимальный тариф
- В случае, если переводить только клиентов с переплатой больше 2000, то падение выручки останется так же в диапазоне 35%


## Подготовка презентационных материалов

https://disk.yandex.ru/client/disk/Проект%20Яндекс

 
