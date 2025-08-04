import os, certifi, nfl_data_py as nfl
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

os.environ["SSL_CERT_FILE"] = certifi.where()

pbp = nfl.import_weekly_data([2019, 2020, 2021, 2022, 2023])

# filter to passes caught by WRs
wr_pass = pbp[(pbp.position == "WR")]

# raw per-game receiving yards
gamelog = (wr_pass.groupby(["player_id", "season", "week"])
                   .agg(rec_yards=("receiving_yards", "sum"),
                        targets=("targets", "count"))
                   .reset_index())

# 3-game rolling features
gamelog = gamelog.sort_values(["player_id", "season", "week"])
gamelog["rec_yards_lag1"] = gamelog.groupby("player_id")["rec_yards"].shift(1)
gamelog["rec_yards_3gm"] = gamelog.groupby("player_id")["rec_yards"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

# simple train-test time split: seasons ≤2022 train, 2023 test
train = gamelog[gamelog.season <= 2022].dropna()
test  = gamelog[gamelog.season == 2023].dropna()

X_train = train[["targets", "rec_yards_lag1", "rec_yards_3gm"]]
y_train = train["rec_yards"]
X_test  = test[["targets", "rec_yards_lag1", "rec_yards_3gm"]]
y_test  = test["rec_yards"]

model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

print("MAE 2023:", mean_absolute_error(y_test, model.predict(X_test)))
print(X_test)
print(y_test)