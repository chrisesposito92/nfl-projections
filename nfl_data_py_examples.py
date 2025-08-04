import nfl_data_py as nfl
import pandas as pd

pd.set_option('display.max_columns', None)

# import_pbp_data() - import play-by-play data
# clean_nfl_data() - clean df by aligning common name diffs
pbp_data = nfl.import_pbp_data([2024])
#print(pbp_data.columns)
pbp_data = nfl.clean_nfl_data(pbp_data)
#print(pbp_data.head(20))

# import_weekly_data() - import weekly player stats
weekly_data = nfl.import_weekly_data([2024])
#print(weekly_data.columns)
weekly_data = nfl.clean_nfl_data(weekly_data)
print(weekly_data.head(20))
