# Imports
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import joblib
import datetime

# silences pandas warnings
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 300)

# Functions
# Function to update Elo rating after a match
def update_elo(player1_elo, player2_elo, result, k_factor=32):
    '''
    This function will take the result from a match and update the elo of the two parties involved
    '''
    expected_score_player1 = 1 / (1 + 10 ** ((player2_elo - player1_elo) / 400))
    expected_score_player2 = 1 - expected_score_player1

    player1_new_elo = player1_elo + k_factor * (result - expected_score_player1)
    player2_new_elo = player2_elo + k_factor * ((1 - result) - expected_score_player2)

    return player1_new_elo, player2_new_elo

# Function to regress Elo ratings towards the mean
def regress_to_mean(elo_ratings, mean_elo, regression_weight=1/3):
    '''
    This function will regress the elo of all parties one third towards the mean. It is meant to be used over the offseason
    '''
    elo_ratings['Elo'] = elo_ratings['Elo'] + regression_weight * (mean_elo - elo_ratings['Elo'])
    return elo_ratings 

# Function to calculate mean Elo
def calculate_mean_elo(elo_ratings):
    return elo_ratings.Elo.sum() / len(elo_ratings)

def get_elo(team,season,week,df):
    '''
    This function will grab the specific elo rating of a team from a specific season and week
    '''
    try:
        elo = [df.loc[(df['Team']==team)&(df['Season']==season)&(df['Week']==week-1)], 'Elo'].values[0]
    except:
        team_week = df.loc[(df['Team']==team)&(df['Season']==season)&(df['Week']<week)]['Week'].max()
        elo = df.loc[(df['Team']==team)&(df['Season']==season)&(df['Week']==team_week), 'Elo'].values[0]   
    return elo

def get_qb_elo(qb_id,season,week,historical_elo_qb):
    '''
    This function will grab the specific elo rating of a QB from a specific season and week
    '''
    if week == 1:
        try:
            qb_week = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==season-1)]['Week'].max()
            elo = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==season-1)&(historical_elo_qb['Week']==qb_week), 'Elo'].values[0]
        except:
            try:
                qb_season = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']<season)]['Season'].max()
                qb_week = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==qb_season)]['Week'].max()
                elo = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==qb_season)&(historical_elo_qb['Week']==qb_week), 'Elo'].values[0]
            except:
                elo = 1500
    else:
        try:
            elo = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==season)&(historical_elo_qb['Week']==week-1), 'Elo'].values[0]
        except:
            try:
                qb_week = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==season)&(historical_elo_qb['Week']<week)]['Week'].max()
                elo = historical_elo_qb.loc[(historical_elo_qb['passer_id']==row['home_qb_id'])&(historical_elo_qb['Season']==season)&(historical_elo_qb['Week']==qb_week), 'Elo'].values[0]
            except:
                try:
                    qb_season = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']<season)]['Season'].max()
                    qb_week = historical_elo_qb.loc[(historical_elo_qb['passer_id']==qb_id)&(historical_elo_qb['Season']==qb_season)]['Week'].max()
                    elo = historical_elo_qb.loc[(historical_elo_qb['passer_id']==row['home_qb_id'])&(historical_elo_qb['Season']==qb_season)&(historical_elo_qb['Week']==qb_week), 'Elo'].values[0]
                except:
                    elo = 1500
    
    return elo

def get_value(df, season, week, team, team_col_name, num, denom):
    '''Get weighted average of stats over the last two seasons'''
    df_temp = df.loc[(df['season']==season)&(df['week']<week)&(df[team_col_name]==team)]
    df_temp2 = df.loc[(df['season']==season-1)&(df[team_col_name]==team)]
    value2 = df_temp2[num].sum()/df_temp2[denom].sum()
    if df_temp[denom].sum() == 0:
        return value2
    value = df_temp[num].sum()/df_temp[denom].sum()
    if week == 1:
        final_value = value2
    elif week > 12:
        final_value = value
    else:
        final_value = week*value/12 + (12-week)*value2/12

    if final_value is None:
        return value2
    else:
        return final_value

def get_qbr(df, season, week, qb_id):
    '''Get weighted average of QBR over the last two seasons'''
    df_temp = df.loc[(df['season']==season)&(df['game_week']<week)&(df['gsis_id']==qb_id)]
    if len(df_temp)>0:
        df_temp['weighted'] = df_temp['qbr_total']*df_temp['qb_plays']/df_temp['qb_plays'].sum()
        value = df_temp['weighted'].sum()
    else:
        value = 50.0
    
    df_temp2 = df.loc[(df['season']==season-1)&(df['gsis_id']==qb_id)]
    if len(df_temp2) > 0:
        df_temp2['weighted'] = df_temp2['qbr_total']*df_temp2['qb_plays']/df_temp2['qb_plays'].sum()
        value2 = df_temp2['weighted'].sum()
    else:
        value2 = 50.0
    if week == 1:
        final_value = value2
    elif week > 12:
        final_value = value
    else:
        final_value = week*value/12 + (12-week)*value2/12
    
    return final_value

# Get the current date
current_date = datetime.datetime.now()

# Extract the year as an integer
current_year = current_date.year

# Last two years for pulling data
years = [current_year, current_year-1]

## Extracting Data from API ##

# IDs
ids = nfl.import_ids()

# Manually imported sheet of starters
starters = pd.read_csv('starting_qbs.csv')

# Weekly data
weekly_data = nfl.import_weekly_data(years)

# Past game data use for elos
matchups = nfl.import_schedules([year for year in range(1999, current_year+1)])
matchups = matchups.loc[~matchups['result'].isna(),:]

# Current season schedule
upcoming = nfl.import_schedules([current_year])
upcoming = upcoming.loc[upcoming['result'].isna(),:]

# QBR
qbr = nfl.import_qbr(years, level='nfl', frequency='weekly')

# play by play data
data = nfl.import_pbp_data([year for year in range(1999, current_year+1)])

## Feature Engineering ##

# Dictionary of all relocated teams
replace_dict = {'OAK':'LV','STL':'LA','SD':'LAC'}

# Replace any relocated teams
data['home_team'] = data['home_team'].replace(replace_dict)
data['away_team'] = data['away_team'].replace(replace_dict)

# Table of per week QB passing performances
pass_att = data.loc[data['play_type_nfl']=='PASS', :]
pass_stats = pass_att.pivot_table(index=['passer_id','passer','season','season_type','week','game_id','game_date','posteam','defteam'], values=['play_id','incomplete_pass','yards_gained','interception', \
    'pass_touchdown'], aggfunc={'play_id':'count', 'incomplete_pass': 'sum','yards_gained': 'sum','interception': 'sum','pass_touchdown': 'sum',}).reset_index().sort_values(by='play_id', \
        ascending=False)

# Create passer rating
pass_stats.loc[:,'completion_pct'] = 1-(pass_stats.loc[:,'incomplete_pass']+pass_stats.loc[:,'interception'])/pass_stats.loc[:,'play_id']
pass_stats.loc[:,'yards_per_att'] = pass_stats.loc[:,'yards_gained']/pass_stats.loc[:,'play_id']
pass_stats.loc[:,'td_per_att'] = pass_stats.loc[:,'pass_touchdown']/pass_stats.loc[:,'play_id']
pass_stats.loc[:,'int_per_att'] = pass_stats.loc[:,'interception']/pass_stats.loc[:,'play_id']
pass_stats.loc[:,'a'] = np.where((5*(pass_stats.loc[:,'completion_pct']-0.3)<=2.375)&(5*(pass_stats.loc[:,'completion_pct']-0.3)>=0), 5*(pass_stats.loc[:,'completion_pct']-0.3), \
    np.where((5*(pass_stats.loc[:,'completion_pct']-0.3)>2.375), 2.375, 0))
pass_stats.loc[:,'b'] = np.where(((0.25*(pass_stats.loc[:,'yards_per_att']-3))<=2.375)&((0.25*(pass_stats.loc[:,'yards_per_att']-3))>=0), (0.25*(pass_stats.loc[:,'yards_per_att']-3)), \
    np.where(((0.25*(pass_stats.loc[:,'yards_per_att']-3))>2.375), 2.375, 0))
pass_stats.loc[:,'c'] = np.where(((20*pass_stats.loc[:,'td_per_att'])<=2.375)&((20*pass_stats.loc[:,'td_per_att'])>=0), (20*pass_stats.loc[:,'td_per_att']), \
    np.where(((20*pass_stats.loc[:,'td_per_att'])>2.375), 2.375, 0))
pass_stats.loc[:,'d'] = np.where(((2.375-(25*pass_stats.loc[:,'int_per_att']))<=2.375)&((2.375-(25*pass_stats.loc[:,'int_per_att']))>=0), (2.375-(25*pass_stats.loc[:,'int_per_att'])), \
    np.where(((2.375-(25*pass_stats.loc[:,'int_per_att']))>2.375), 2.375, 0))
pass_stats.loc[:,'passer_rating'] = 100*((pass_stats.loc[:,'a']+pass_stats.loc[:,'b']+pass_stats.loc[:,'c']+pass_stats.loc[:,'d'])/6)

# At least 10 pass attempts
pass_stats = pass_stats.loc[pass_stats['play_id']>=10]
pass_stats = pass_stats.sort_values(by='game_date').reset_index()

# To check for season change
pass_stats['season_shift'] = pass_stats['season'].shift().fillna(0.0)

# Initial elo rating
initial_elo = 1500

# Create a dictionary to hold current Elo ratings for each player
qb_ratings = {player: initial_elo for player in pass_stats['passer_id'].unique()}
def_ratings = {team: initial_elo for team in pass_stats['defteam'].unique()}

# Create a separate DataFrame to store the updated Elo ratings
elo_qb = pd.DataFrame(qb_ratings.items(), columns=['passer_id', 'Elo'])
elo_qb = elo_qb.merge(pass_stats[['passer_id', 'passer']].drop_duplicates(), how='inner', on='passer_id')
elo_def = pd.DataFrame(def_ratings.items(), columns=['Team', 'Elo'])

# Create a new DataFrame to hold historical weekly Elo ratings for each team and QB
historical_elo_def = pd.DataFrame(columns=['Team', 'Season', 'Week', 'Elo'])
historical_elo_qb = pd.DataFrame(columns=['passer_id', 'Passer', 'Season', 'Week', 'Elo'])

# Iterate through matchups DataFrame and update Elo ratings
for index, row in pass_stats.iterrows():

    # Check if the season has ended (you need to define the condition for season end)
    if (row['season_shift']!=0.0) & (row['season'] != row['season_shift']):
        
        # Calculate mean Elo at the end of the season
        mean_elo = calculate_mean_elo(elo_def)

        # Regress each team's Elo ratings towards the mean
        elo_def = regress_to_mean(elo_def, mean_elo, regression_weight=1/3)
        
        elo_def_temp = elo_def.copy()
        
        elo_def_temp['Week'] = 0
        elo_def_temp['Season'] = row['season']
        
        historical_elo_def = pd.concat([historical_elo_def,elo_def_temp])
    
    player1 = row['passer_id']
    player2 = row['defteam']
    result = 1 if row['passer_rating'] >= pass_stats.loc[pass_stats.season==row['season']].passer_rating.median() else 0 ## A passer rating above season median is considered a win for the QB

    player1_elo = elo_qb.loc[elo_qb['passer_id']==row['passer_id']].reset_index().loc[0,'Elo']
    player2_elo = elo_def.loc[elo_def['Team']==row['defteam']].reset_index().loc[0,'Elo']

    player1_new_elo, player2_new_elo = update_elo(player1_elo, player2_elo, result)
    
    elo_qb.loc[elo_qb['passer_id'] == row['passer_id'], 'Elo'] = player1_new_elo
    elo_def.loc[elo_def['Team'] == row['defteam'], 'Elo'] = player2_new_elo
    
    # Append the updated Elo ratings to the historical Elo DataFrame
    historical_elo_qb = pd.concat([
        historical_elo_qb,
        pd.DataFrame([{'passer_id': row['passer_id'], 'Passer': row['passer'], 'Season': row['season'], 'Week': row['week'], 'Elo': player1_new_elo}])
    ])
    historical_elo_def = pd.concat([
        historical_elo_def,
        pd.DataFrame([{'Team': row['defteam'], 'Season': row['season'], 'Week': row['week'], 'Elo': player2_new_elo}])
    ])

# Create a dictionary to hold current Elo ratings for each team
elo_ratings = {team: initial_elo for team in pd.concat([matchups['away_team'], matchups['home_team']]).unique()}

# Create a separate DataFrame to store the updated Elo ratings
elo_df = pd.DataFrame(elo_ratings.items(), columns=['Team', 'Elo'])

# Replace team names in past matchups
matchups['away_team'] = matchups['away_team'].replace(replace_dict)
matchups['home_team'] = matchups['home_team'].replace(replace_dict)

# To check for season change
matchups['season_shift'] = matchups['season'].shift().fillna(0.0)

# Create a new DataFrame to hold historical weekly Elo ratings for each team
historical_elo_df = pd.DataFrame(columns=['Team', 'Season', 'Week', 'Elo'])

# Iterate through matchups DataFrame and update Elo ratings
for index, row in matchups.iterrows():
    # Check if the season has ended (you need to define the condition for season end)
    if (row['season_shift']!=0.0) & (row['season'] != row['season_shift']):
        
        # Calculate mean Elo at the end of the season
        mean_elo = calculate_mean_elo(elo_df)

        # Regress each team's Elo ratings towards the mean
        elo_df = regress_to_mean(elo_df, mean_elo, regression_weight=1/3)
        
        elo_df_temp = elo_df.copy()
        
        elo_df_temp['Week'] = 0
        elo_df_temp['Season'] = row['season']
        
        historical_elo_df = pd.concat([historical_elo_df,elo_df_temp])
    
    if row['result'] < 0:
        result = 1
    elif row['result'] == 0:
        result = 0.5
    else:
        result = 0

    player1_elo = elo_df.loc[elo_df['Team']==row['away_team']].reset_index().loc[0,'Elo']
    player2_elo = elo_df.loc[elo_df['Team']==row['home_team']].reset_index().loc[0,'Elo']

    player1_new_elo, player2_new_elo = update_elo(player1_elo, player2_elo, result)
    
    # Update the main Elo DataFrame with the updated Elo ratings
    elo_df.loc[elo_df['Team'] == row['away_team'], 'Elo'] = player1_new_elo
    elo_df.loc[elo_df['Team'] == row['home_team'], 'Elo'] = player2_new_elo
    
    # Append the updated Elo ratings to the historical Elo DataFrame
    historical_elo_df = pd.concat([
        historical_elo_df,
        pd.DataFrame([{'Team': row['away_team'], 'Season': row['season'], 'Week': row['week'], 'Elo': player1_new_elo}]),
        pd.DataFrame([{'Team': row['home_team'], 'Season': row['season'], 'Week': row['week'], 'Elo': player2_new_elo},])
    ])

# Replace team names
weekly_data['recent_team'] = weekly_data['recent_team'].replace(replace_dict)

# Pivot table of weekly team metrics
weekly_sum = weekly_data.pivot_table(index=['season','week','recent_team'], values=['carries','rushing_yards','rushing_epa',\
    'passing_epa','attempts','sacks'], aggfunc='sum').reset_index().sort_values(by=['season','week'], ascending=False)

# Identify if there was a turnover in a drive/ how many points scored
data['drive_turnover'] = np.where(data['fixed_drive_result'].isin(['Turnover','Opp touchdown']), 1.0, 0.0)
data['drive_points'] = np.where(data['fixed_drive_result']=='Touchdown', 6.0, np.where(data['fixed_drive_result']=='Field goal', 3.0, np.where(data['fixed_drive_result']=='Safety', -2.0, 0.0)))

# Drop any NA plays
drive_df = data.loc[~data['drive_play_count'].isna()].drop_duplicates(subset=['game_id','fixed_drive'], keep='first')

# Grab yardage, 3rd and 4th down conversions, drive success rates, epa, drive turnover rates, redzone conversion
off_yardage = data.pivot_table(index=['season','week','posteam','defteam'], values=['yards_gained','play','third_down_converted','third_down_failed','fourth_down_converted','fourth_down_failed',\
    'epa'], aggfunc='sum').reset_index().sort_values(by=['season','week'], ascending=False)
drive_data = drive_df.pivot_table(index=['season','week','posteam','defteam'], values=['fixed_drive','drive_points','drive_turnover','drive_play_count','drive_first_downs','drive_inside20',\
    'drive_yards_penalized'], aggfunc={'fixed_drive':'count','drive_points': 'sum', 'drive_turnover': 'sum', 'drive_play_count': 'sum', 'drive_first_downs':'sum', 'drive_inside20':'sum',\
        'drive_yards_penalized':'sum'}).reset_index().sort_values(by=['season','week'], ascending=False)
rz_data = drive_df.loc[drive_df['drive_inside20']==1].pivot_table(index=['season','week','posteam','defteam'], values=['drive_inside20','drive_points'], aggfunc='sum').reset_index().sort_values(by=[\
    'season','week'], ascending=False)

# Total third and fourth downs
off_yardage['third_down_total'] = off_yardage['third_down_converted'] + off_yardage['third_down_failed']
off_yardage['fourth_down_total'] = off_yardage['fourth_down_converted'] + off_yardage['fourth_down_failed']

# Merge in passer id
qbr['player_id'] = qbr['player_id'].astype(float)
ids['espn_id'] = ids['espn_id'].astype(float)
qbr = qbr.merge(ids[['espn_id','gsis_id']], how='left', left_on ='player_id', right_on='espn_id')

# Replace team names
qbr['team_abb'] = qbr['team_abb'].replace(replace_dict)

## Preparing Predictions df ##

# merge in starting QB
upcoming = upcoming.drop(['away_qb_id','home_qb_id','away_qb_name','home_qb_name'],axis=1)

upcoming = upcoming.merge(starters[['Team','passer','passer_id']], how='left', left_on='home_team', right_on='Team').rename({'passer':\
    'home_qb_name', 'passer_id':'home_qb_id'},axis=1)

upcoming = upcoming.drop(['Team'],axis=1)

upcoming = upcoming.merge(starters[['Team','passer','passer_id']], how='left', left_on='away_team', right_on='Team').rename({'passer':\
    'away_qb_name', 'passer_id':'away_qb_id'},axis=1)

upcoming = upcoming.drop(['Team'],axis=1)

# grab a main df of all important info
newseason = upcoming[['game_id','season','week','away_team','away_score','home_team','home_score','result','location','total','away_rest','home_rest','away_moneyline',\
    'home_moneyline','spread_line','total_line','div_game','roof','surface','away_qb_id','home_qb_id','away_qb_name','home_qb_name']]

# Loop through upcoming games and populate cols (features individually)
newseason.loc[:,'home_elo'] = np.nan
newseason.loc[:,'away_elo'] = np.nan
newseason.loc[:,'home_pass_elo_off'] = np.nan # QB elo Def elo difference
newseason.loc[:,'away_pass_elo_off'] = np.nan # QB elo Def elo difference
newseason.loc[:,'home_pass_elo_def'] = np.nan # QB elo Def elo difference
newseason.loc[:,'away_pass_elo_def'] = np.nan # QB elo Def elo difference
newseason.loc[:,'home_rush_ypc'] = np.nan
newseason.loc[:,'away_rush_ypc'] = np.nan
newseason.loc[:,'home_rush_epa_play'] = np.nan
newseason.loc[:,'away_rush_epa_play'] = np.nan
newseason.loc[:,'home_qbr'] = np.nan
newseason.loc[:,'away_qbr'] = np.nan
newseason.loc[:,'home_epa_play'] = np.nan 
newseason.loc[:,'away_epa_play'] = np.nan 
newseason.loc[:,'home_epa_play_def'] = np.nan
newseason.loc[:,'away_epa_play_def'] = np.nan
newseason.loc[:,'home_yds_play'] = np.nan
newseason.loc[:,'away_yds_play'] = np.nan
newseason.loc[:,'home_yds_play_def'] = np.nan
newseason.loc[:,'away_yds_play_def'] = np.nan
newseason.loc[:,'home_3d_conv'] = np.nan
newseason.loc[:,'away_3d_conv'] = np.nan
newseason.loc[:,'home_3d_conv_def'] = np.nan
newseason.loc[:,'away_3d_conv_def'] = np.nan
newseason.loc[:,'home_4d_conv'] = np.nan
newseason.loc[:,'away_4d_conv'] = np.nan
newseason.loc[:,'home_4d_conv_def'] = np.nan
newseason.loc[:,'away_4d_conv_def'] = np.nan
newseason.loc[:,'home_1D_drive'] = np.nan
newseason.loc[:,'away_1D_drive'] = np.nan
newseason.loc[:,'home_1D_drive_def'] = np.nan
newseason.loc[:,'away_1D_drive_def'] = np.nan
newseason.loc[:,'home_RZ_drive'] = np.nan
newseason.loc[:,'away_RZ_drive'] = np.nan
newseason.loc[:,'home_RZ_drive_def'] = np.nan
newseason.loc[:,'away_RZ_drive_def'] = np.nan
newseason.loc[:,'home_play_drive'] = np.nan
newseason.loc[:,'away_play_drive'] = np.nan
newseason.loc[:,'home_play_drive_def'] = np.nan
newseason.loc[:,'away_play_drive_def'] = np.nan
newseason.loc[:,'home_points_drive'] = np.nan
newseason.loc[:,'away_points_drive'] = np.nan
newseason.loc[:,'home_points_drive_def'] = np.nan
newseason.loc[:,'away_points_drive_def'] = np.nan
newseason.loc[:,'home_to_drive'] = np.nan
newseason.loc[:,'away_to_drive'] = np.nan
newseason.loc[:,'home_to_drive_def'] = np.nan
newseason.loc[:,'away_to_drive_def'] = np.nan
newseason.loc[:,'home_pen_yds_drive'] = np.nan
newseason.loc[:,'away_pen_yds_drive'] = np.nan
newseason.loc[:,'home_pen_yds_drive_def'] = np.nan
newseason.loc[:,'away_pen_yds_drive_def'] = np.nan
newseason.loc[:,'home_points_RZ'] = np.nan
newseason.loc[:,'away_points_RZ'] = np.nan
newseason.loc[:,'home_points_RZ_def'] = np.nan
newseason.loc[:,'away_points_RZ_def'] = np.nan

# Change dtypes
newseason['season'] = newseason['season'].astype(int)
newseason['week'] = newseason['week'].astype(int)
historical_elo_df['Season'] = historical_elo_df['Season'].astype(int)
historical_elo_df['Week'] = historical_elo_df['Week'].astype(int)

newseason = newseason.reset_index(drop=True)

current_week = newseason.loc[0,'week']

for i,row in newseason.iterrows():
    # Populate elo differences
    newseason.loc[i,'home_elo'], newseason.loc[i,'away_elo'] = get_elo(row['home_team'],row['season'],current_week,historical_elo_df), get_elo(row['away_team'],row['season'],current_week,historical_elo_df)
        
    newseason.loc[i,'home_pass_elo_off'] = get_qb_elo(row['home_qb_id'],row['season'],current_week,historical_elo_qb) 
    newseason.loc[i,'away_pass_elo_off'] = get_qb_elo(row['away_qb_id'],row['season'],current_week,historical_elo_qb)
    
    newseason.loc[i,'home_pass_elo_def'], newseason.loc[i,'away_pass_elo_def'] = get_elo(row['home_team'],row['season'],current_week,historical_elo_def), get_elo(row['away_team'],row['season'],current_week,historical_elo_def)
    
    # Populate other stats
    newseason.loc[i,'home_rush_ypc'] = get_value(weekly_sum, row['season'], current_week, row['home_team'], 'recent_team', 'rushing_yards', \
        'carries')
    newseason.loc[i,'away_rush_ypc'] = get_value(weekly_sum, row['season'], \
        current_week, row['away_team'], 'recent_team', 'rushing_yards', 'carries')
    
    newseason.loc[i,'home_rush_epa_play'] = get_value(weekly_sum, row['season'], current_week, row['home_team'], 'recent_team', 'rushing_epa', \
        'carries') 
    newseason.loc[i,'away_rush_epa_play'] = get_value(weekly_sum, row['season'], \
        current_week, row['away_team'], 'recent_team', 'rushing_epa', 'carries')
    
    newseason.loc[i,'home_qbr'] = get_qbr(qbr, row['season'], current_week, row['home_qb_id'])
    newseason.loc[i,'away_qbr'] = get_qbr(qbr, row['season'], current_week, row['away_qb_id'])
    
    newseason.loc[i,'home_epa_play'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'posteam', 'epa', 'play')
    newseason.loc[i,'away_epa_play'] = get_value(off_yardage, row['season'], current_week, row['away_team'], 'posteam', 'epa', 'play')
    
    newseason.loc[i,'home_epa_play_def'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'defteam', 'epa', 'play')
    newseason.loc[i,'away_epa_play_def'] = get_value(off_yardage, row['season'], current_week, row['away_team'], 'defteam', 'epa', 'play')
    
    newseason.loc[i,'home_yds_play'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'posteam', 'yards_gained', 'play')
    newseason.loc[i,'away_yds_play'] = get_value(off_yardage, row['season'], current_week, row['away_team'], 'posteam', 'yards_gained', 'play')
    
    newseason.loc[i,'home_yds_play_def'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'defteam', 'yards_gained', 'play')
    newseason.loc[i,'away_yds_play_def']= get_value(off_yardage, row['season'], current_week, row['away_team'], 'defteam', 'yards_gained', 'play')
    
    newseason.loc[i,'home_3d_conv'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'posteam', 'third_down_converted', \
                                             'third_down_total')
    newseason.loc[i,'away_3d_conv'] = get_value(off_yardage, row['season'], \
        current_week, row['away_team'], 'posteam', 'third_down_converted', 'third_down_total')
    
    newseason.loc[i,'home_3d_conv_def'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'defteam', 'third_down_converted',\
                                                 'third_down_total') 
    newseason.loc[i,'away_3d_conv_def'] = get_value(off_yardage, row['season'], \
        current_week, row['away_team'], 'defteam', 'third_down_converted', 'third_down_total')
    
    newseason.loc[i,'home_4d_conv'] = get_value(off_yardage, row['season'], current_week, row['home_team'], 'posteam', 'fourth_down_converted', \
                                             'fourth_down_total')
    newseason.loc[i,'away_4d_conv'] = get_value(off_yardage, row['season'], \
        current_week, row['away_team'], 'posteam', 'fourth_down_converted', 'fourth_down_total')
    
    newseason.loc[i,'home_4d_conv_def'] = get_value(off_yardage, row['season'],current_week, row['home_team'], 'defteam', \
                                                 'fourth_down_converted', 'fourth_down_total') 
    newseason.loc[i,'away_4d_conv_def'] = get_value(off_yardage, row['season'], \
        current_week, row['away_team'], 'defteam', 'fourth_down_converted', 'fourth_down_total')
    
    newseason.loc[i,'home_1D_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_first_downs', \
                                              'fixed_drive')
    newseason.loc[i,'away_1D_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_first_downs', 'fixed_drive')
    
    newseason.loc[i,'home_1D_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_first_downs', \
                                                  'fixed_drive')
    newseason.loc[i,'away_1D_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_first_downs', 'fixed_drive')
    
    newseason.loc[i,'home_RZ_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_inside20', \
                                              'fixed_drive')
    newseason.loc[i,'away_RZ_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_inside20', 'fixed_drive')
    
    newseason.loc[i,'home_RZ_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_inside20', \
                                                  'fixed_drive')
    newseason.loc[i,'away_RZ_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_inside20', 'fixed_drive')
    
    newseason.loc[i,'home_play_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_play_count', \
                                                'fixed_drive')
    newseason.loc[i,'away_play_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_play_count', 'fixed_drive')
    
    newseason.loc[i,'home_play_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_play_count', \
                                                    'fixed_drive')
    newseason.loc[i,'away_play_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_play_count', 'fixed_drive')
    
    newseason.loc[i,'home_points_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_points', \
                                                  'fixed_drive')
    newseason.loc[i,'away_points_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_points', 'fixed_drive')
    
    newseason.loc[i,'home_points_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_points', \
                                                      'fixed_drive')
    newseason.loc[i,'away_points_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_points', 'fixed_drive')
    
    newseason.loc[i,'home_to_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_turnover', \
                                                   'fixed_drive')
    newseason.loc[i,'away_to_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_turnover', 'fixed_drive')
    
    newseason.loc[i,'home_to_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_turnover',\
                                                   'fixed_drive')
    newseason.loc[i,'away_to_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_turnover', 'fixed_drive')
    
    newseason.loc[i,'home_pen_yds_drive'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'posteam', \
                                                   'drive_yards_penalized', 'fixed_drive')
    newseason.loc[i,'away_pen_yds_drive'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_yards_penalized', 'fixed_drive')
    
    newseason.loc[i,'home_pen_yds_drive_def'] = get_value(drive_data, row['season'], current_week, row['home_team'], 'defteam', \
                                                       'drive_yards_penalized', 'fixed_drive')
    newseason.loc[i,'away_pen_yds_drive_def'] = get_value(drive_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_yards_penalized', 'fixed_drive')
    
    newseason.loc[i,'home_points_RZ'] = get_value(rz_data, row['season'], current_week, row['home_team'], 'posteam', 'drive_points', \
                                               'drive_inside20')
    newseason.loc[i,'away_points_RZ'] = get_value(rz_data, row['season'], \
        current_week, row['away_team'], 'posteam', 'drive_points', 'drive_inside20')
    
    newseason.loc[i,'home_points_RZ_def'] = get_value(rz_data, row['season'], current_week, row['home_team'], 'defteam', 'drive_points', \
                                                   'drive_inside20')
    newseason.loc[i,'away_points_RZ_def'] = get_value(rz_data, row['season'], \
        current_week, row['away_team'], 'defteam', 'drive_points', 'drive_inside20')
    
# Create a col if game is in dome
newseason.loc[:,'is_dome'] = np.where(newseason['roof']=='dome', 1.0, 0.0)

# Create a col if game is played on natural grass
newseason.loc[:,'is_grass'] = np.where(newseason['surface']=='grass',1.0,0.0)

# Create a col if game is played at neutral site
newseason.loc[:,'is_neutral'] = np.where(newseason['location']=='Neutral',1.0,0.0)

## Making Predictions ##

# Loading model
mdl = joblib.load('NFL_2023_game_prediction.jlb')

# List of features used in model
features = ['away_rest','home_rest','spread_line','total_line','div_game','home_elo', 'away_elo','home_pass_elo_off','away_pass_elo_off',
            'home_pass_elo_def','away_pass_elo_def','home_rush_ypc','away_rush_ypc','home_rush_epa_play','away_rush_epa_play','home_qbr','away_qbr',
            'home_epa_play','away_epa_play','home_epa_play_def','away_epa_play_def','home_yds_play','away_yds_play','home_yds_play_def',
            'away_yds_play_def','home_3d_conv','away_3d_conv','home_3d_conv_def','away_3d_conv_def','home_4d_conv',
            'away_4d_conv','home_4d_conv_def','away_4d_conv_def','home_1D_drive','away_1D_drive','home_1D_drive_def','away_1D_drive_def',
            'home_points_drive','away_points_drive','home_points_drive_def','away_points_drive_def','home_to_drive',
            'away_to_drive','home_to_drive_def','away_to_drive_def','home_pen_yds_drive','away_pen_yds_drive','home_pen_yds_drive_def',
            'away_pen_yds_drive_def','home_points_RZ','away_points_RZ','home_points_RZ_def','away_points_RZ_def','is_dome','is_grass']

# Prediction dataframe
pred_data = newseason[sorted(features)]

# Make predictions
y_proba = mdl.predict_proba(pred_data)

# Attach predictions back into main df
newseason['home_team_win_prob'] = y_proba[:,1]

# Output all info
newseason.to_csv(f'NFL_{current_year}_{current_week}_predictions.csv')
historical_elo_def.to_csv('def_elos.csv')
historical_elo_df.to_csv('team_elos.csv')
historical_elo_qb.to_csv('qb_elos.csv')