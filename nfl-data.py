import nfl_data_py as nfl
import pandas as pd
from xgboost import XGBRegressor

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns

# 1. Data pull
weekly_data = nfl.import_weekly_data([2024])

weekly_data = nfl.clean_nfl_data(weekly_data)

# Filter
weekly_data = weekly_data[weekly_data['player_display_name'] == "Breece Hall"]
#print(weekly_data.columns)

players = nfl.import_players()
players = nfl.clean_nfl_data(players)
wr_active = players.query("position=='WR' and last_season==2025 and espn_id.notna()").copy()
latest_team_grouped = wr_active.groupby("latest_team").agg({"gsis_id": "count"}).reset_index()
#print(latest_team_grouped)
#print(players.columns)

#print(weekly_data)

seasonal_data = nfl.import_seasonal_data([2024])
seasonal_data = nfl.clean_nfl_data(seasonal_data)

# bring in the display_name from players where player_id matches gsis_id from players
seasonal_data = seasonal_data.merge(players[['gsis_id', 'position', 'display_name']], left_on='player_id', right_on='gsis_id', how='left')

# Filter
#seasonal_data = seasonal_data[seasonal_data['player_display_name'] == "Breece Hall"]

# Sort by receiving_epa descending
seasonal_data = seasonal_data.sort_values(by='passing_epa', ascending=False).head(10)

#print(seasonal_data.columns)

ngs = nfl.import_ngs_data('receiving', [2024])
ngs = ngs[ngs['player_last_name'] == "Chase"]
#print(ngs.columns)
#print(ngs)

schedules = nfl.import_schedules([2025])
print(schedules)

weekly_rosters = nfl.import_weekly_rosters([2025])
print(weekly_rosters)