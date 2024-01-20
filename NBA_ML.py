import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def add_target(team):
    team['target'] = team['won'].shift(-1)
    return team
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    all_seasons = sorted(data['season'].unique())
    
    for i in range(start, len(all_seasons), step):
        season = all_seasons[i]
        
        train = data[data['season']<season]
        test  = data[data['season'] == season]
        
        model.fit(train[predictors], train['target'])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        
        combined = (pd.concat([test['target'], preds], axis=1)).rename(columns={'target':'Result', 0:'Prediction'})
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

def team_won(x):
   return x[x['won'] == 1].shape[0]/x.shape[0]

def team_average(team):
    average = team.rolling(10).mean()
    return average

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(dataframe, col):
    return dataframe.groupby('team', group_keys = False).apply(lambda x: shift_col(x, col))

stats = pd.read_csv("nba_games.csv", index_col = 0).sort_values('date').reset_index(drop=True)
stats = stats.groupby('team', group_keys=False).apply(add_target)
stats['target'][pd.isnull(stats['target'])] = 2
stats['target'] = stats['target'].astype(int, errors='ignore')

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits = 3)

sfs = SequentialFeatureSelector(rr, n_features_to_select = 30, direction = 'forward', cv=split)

removed_columns = ['date', 'team', 'team_opp', 'won', 'target', 'season', 'mp.1', 'mp_opp.1', 'index_opp',
                  'mp_max', '+/-_opp', 'mp_max_opp', '+/-', 'mp_max.1', 'mp_max_opp.1']
removed_columns2 = ['mp.1', 'mp_opp.1', 'index_opp',
                  'mp_max', '+/-_opp', 'mp_max_opp', '+/-', 'mp_max.1', 'mp_max_opp.1']
selected_columns = stats.columns[~stats.columns.isin(removed_columns)]
selected_columns2 = stats.columns[~stats.columns.isin(removed_columns2)]

scaler = MinMaxScaler()

stats.groupby('home').apply(team_won)
stats_roll = stats[list(selected_columns) + ['won','team','season']] 
stats_roll = stats_roll.groupby(['team', 'season'], group_keys=False).apply(team_average)

roll_cols = [f"{col}_10" for col in stats_roll.columns]
stats_roll.columns = roll_cols

stats_roll.columns = roll_cols
stats = pd.concat([stats[selected_columns2], stats_roll], axis=1).dropna()

stats['home_next'] = add_col(stats, 'home')
stats['team_opp_next'] = add_col(stats, 'team_opp')
stats['date_next'] = add_col(stats, 'date')

full = stats.merge(stats[roll_cols + ['team_opp_next','date_next','team']],
                   left_on=['team','date_next'],
                   right_on=['team_opp_next', 'date_next']
                   )
removed_columns = list(full.columns[full.dtypes=='object']) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
sfs.fit(full[selected_columns], full['target'])
predictor = list(selected_columns[sfs.get_support()])
ML_Prediction = backtest(full, rr, predictor)
print(accuracy_score(ML_Prediction['Result'], ML_Prediction['Prediction']))
