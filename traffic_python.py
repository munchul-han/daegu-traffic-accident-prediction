import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
plt.rcParams["font.family"] = "Malgun Gothic"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # pythonhashseed 환경변수 설정
    np.random.seed(seed)


def process_location_data(df, address_column, drop_columns):
    location_pattern = r"(\S+) (\S+) (\S+) (\S+)"
    df[["도시", "구", "동", "우편번호"]] = df[address_column].str.extract(location_pattern)
    df = df.drop(drop_columns, axis=1)
    return df


def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataframe.columns
    vif_data["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    vif_data["VIF"] = vif_data["VIF"].apply(lambda x: f"{x:.2f}")
    return vif_data


seed_everything(42)


path = "C:/Users/yis82/OneDrive/Desktop/daegu-traffic-accident-prediction/data"
fpath = "C:/Users/yis82/OneDrive/Desktop/daegu-traffic-accident-prediction/data/external_open"
train_data = pd.read_csv(path + "/train.csv")
test_data = pd.read_csv(path + "/test.csv")
# acc = pd.read_csv(fpath + "/countrywide_accident.csv")
CCTV = pd.read_csv(fpath + "/대구 CCTV 정보.csv", encoding="cp949")
light = pd.read_csv(fpath + "/대구 보안등 정보.csv", encoding="cp949", low_memory=False)
child_protect = pd.read_csv(fpath + "/대구 어린이 보호 구역 정보.csv", encoding="cp949")
parking = pd.read_csv(fpath + "/대구 주차장 정보.csv", encoding="cp949")
train = train_data.copy()
test = test_data.copy()
CCTV = process_location_data(CCTV, "소재지지번주소", ["소재지지번주소", "우편번호"])
light = process_location_data(light, "소재지지번주소", ["소재지지번주소", "우편번호"])
parking = process_location_data(parking, "소재지지번주소", ["소재지지번주소", "우편번호"])
child_protect = process_location_data(child_protect, "소재지지번주소", ["소재지지번주소", "우편번호"])

CCTV_notnull = CCTV.dropna(subset=["동"])
light_notnull = light.dropna(subset=["동"])
parking_notnull = parking.dropna(subset=["동"])
child_protect_notnull = child_protect.dropna(subset=["동"])

light_df = light_notnull.groupby("동").size().reset_index(name="보안등개수")
CCTV_num = CCTV_notnull.groupby("동").size().reset_index(name="CCTV개수")
CCTV_pre = CCTV_notnull[["동", "단속구분"]].groupby(["동", "단속구분"]).size().unstack()
CCTV_cat = CCTV_pre.add_prefix("단속구분_")
CCTV_cat = CCTV_cat.fillna(0)
CCTV_df = pd.merge(CCTV_num, CCTV_cat, how="left", on="동")
parking_cat = pd.get_dummies(
    parking_notnull[["동", "급지구분"]], columns=["급지구분"], prefix="급지"
)
parking_num = parking_notnull.groupby("동").size().reset_index(name="주차장개수")
parking_df = pd.merge(parking_num, parking_cat, how="left", on="동")
elementary_num = child_protect_notnull.groupby("동")["대상시설명"].nunique()
elementary_df = pd.DataFrame(elementary_num).reset_index()
elementary_df.rename(columns={"대상시설명": "초등학교개수"}, inplace=True)
cols = ["ID", "사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형", "ECLO"]
train = train[cols]
train.isna().sum()


train["사고일시"] = train["사고일시"].astype("datetime64[ns]")
train["사고월"] = train["사고일시"].dt.month.astype(str)
train["사고일"] = train["사고일시"].dt.day.astype(str)
train["사고시간"] = train["사고일시"].dt.hour.astype(str)

train = train.drop("사고일시", axis=1)

location_pattern = r"(\S+) (\S+) (\S+)"
train[["도시", "구", "동"]] = train["시군구"].str.extract(location_pattern)
train = train.drop(columns=["시군구"])

# test도 train처럼 맞춰주기

test["사고일시"] = test["사고일시"].astype("datetime64[ns]")
test["사고월"] = test["사고일시"].dt.month.astype(str)
test["사고일"] = test["사고일시"].dt.day.astype(str)
test["사고시간"] = test["사고일시"].dt.hour.astype(str)

test = test.drop("사고일시", axis=1)

location_pattern = r"(\S+) (\S+) (\S+)"
test[["도시", "구", "동"]] = test["시군구"].str.extract(location_pattern)
test = test.drop(columns=["시군구"])

join_1 = pd.merge(train, CCTV_df, on="동", how="left")
join_2 = pd.merge(join_1, light_df, on="동", how="left")
join_3 = pd.merge(join_2, parking_df, on="동", how="left")
join_3 = join_3.drop_duplicates("ID")
join_4 = pd.merge(join_3, elementary_df, on="동", how="left")
train_raw = join_4.fillna(0).copy()

join_1 = pd.merge(test, CCTV_df, on="동", how="left")
join_2 = pd.merge(join_1, light_df, on="동", how="left")
join_3 = pd.merge(join_2, parking_df, on="동", how="left")
join_3 = join_3.drop_duplicates("ID")
join_4 = pd.merge(join_3, elementary_df, on="동", how="left")
test_raw = join_4.fillna(0).copy()

invalid_dongs = train_raw[~train_raw["동"].isin(test_raw["동"])]
train_raw = train_raw[~train_raw["동"].isin(invalid_dongs["동"])]

train_cat = pd.get_dummies(
    data=train_raw, columns=["요일", "기상상태", "도로형태", "노면상태", "사고유형", "구"]
)
train_cat = train_cat.drop(["도시"], axis=1)

test_cat = pd.get_dummies(
    data=test_raw, columns=["요일", "기상상태", "도로형태", "노면상태", "사고유형", "구"]
)
test_cat = test_cat.drop(["도시"], axis=1)

train_col = list(train_cat.columns)
test_col = list(test_cat.columns)
list(set(train_col) - set(test_col))

foggy = [0] * len(test_cat)
test_cat["기상상태_안개"] = foggy

train_cat[["사고월", "사고일", "사고시간"]] = train_cat[["사고월", "사고일", "사고시간"]].apply(
    pd.to_numeric
)
test_cat[["사고월", "사고일", "사고시간"]] = test_cat[["사고월", "사고일", "사고시간"]].apply(pd.to_numeric)

train_col = list(train_cat.columns)
test_col = list(test_cat.columns)
list(set(train_col) - set(test_col))

train_raw.drop_duplicates(inplace=True)

test_x = test_raw.drop(columns=["ID"]).copy()
train_x = train_raw[test_x.columns].copy()
train_y = train_raw["ECLO"].copy()

ohe = OneHotEncoder(sparse=False)
train_gu_name = ohe.fit_transform(train_x[["구"]])


categorical_features = list(
    train_x.dtypes[train_x.dtypes == "object"].index
)  # object값 list에 넣음
# 추출된 문자열 변수 확인

for i in categorical_features:  # 인코딩 적용한 값 반환
    le = TargetEncoder(cols=[i])
    train_x[i] = le.fit_transform(train_x[i], train_y)
    test_x[i] = le.transform(test_x[i])


# target encdoer 주의사항: train은 fit_transform인 반면, test는 transform만 진행!

vif_df = calculate_vif(train_x.iloc[:, :-8])  # one-hot encoding 값은 제거하고 독립변수끼리만 확인
print(vif_df)
