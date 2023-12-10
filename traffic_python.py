import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "C:/Users/yis82/OneDrive/Desktop/daegu-traffic-accident-prediction/data"
fpath = "C:/Users/yis82/OneDrive/Desktop/daegu-traffic-accident-prediction/data/external_open"
npath = (
    "C:/Users/yis82/OneDrive/Desktop/daegu-traffic-accident-prediction/data/new_data"
)


def process_location_data(df, address_column, drop_columns):
    location_pattern = r"(\S+) (\S+) (\S+) (\S+)"
    df[["도시", "구", "동", "우편번호"]] = df[address_column].str.extract(location_pattern)
    df = df.drop(drop_columns, axis=1)
    return df


train_data = pd.read_csv(path + "/train.csv")
test_data = pd.read_csv(path + "/test.csv")
CCTV = pd.read_csv(fpath + "/대구 CCTV 정보.csv", encoding="cp949")
light = pd.read_csv(fpath + "/대구 보안등 정보.csv", encoding="cp949", low_memory=False)
child_protect = pd.read_csv(fpath + "/대구 어린이 보호 구역 정보.csv", encoding="cp949")
parking = pd.read_csv(fpath + "/대구 주차장 정보.csv", encoding="cp949")
crossboard = pd.read_excel(npath + "/법정동별_횡단보도수.xlsx")
numcar = pd.read_csv(npath + "/대구광역시_읍면동별 자동차 등록현황_20211031.csv", encoding="cp949")
senior_center = pd.read_csv(npath + "/대구_경로당현황_법정동_20211110.csv", encoding="euc-kr")
train = train_data.copy()
test = test_data.copy()
time_pattern = r"(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})"
# 날짜와 시간을 추출하기 위한 정규표현식
# \d{4}는 연도(4자리 숫자), \d{1,2}는 월과 일(1자리 또는 2자리 숫자), 그리고 \d{1,2}는 시간(1자리 또는 2자리 숫자)

train[["연", "월", "일", "시간"]] = train["사고일시"].str.extract(time_pattern)
# pd.to_numeric함수는 숫자처럼 보이는 문자들을 숫자로 바꿔줌
train[["연", "월", "일", "시간"]] = train[["연", "월", "일", "시간"]].apply(
    pd.to_numeric
)  # 추출된 문자열을 수치화
train = train.drop(columns=["사고일시"])  # 정보 추출이 완료된 '사고일시' 컬럼은 제거

location_pattern = r"(\S+) (\S+) (\S+)"
train[["도시", "구", "동"]] = train["시군구"].str.extract(location_pattern)
train = train.drop(columns=["시군구"])
# test도 train처럼 맞춰주기

# 해당 과정을 test_x에 대해서도 반복해줍니다
test[["연", "월", "일", "시간"]] = test["사고일시"].str.extract(time_pattern)
test[["연", "월", "일", "시간"]] = test[["연", "월", "일", "시간"]].apply(pd.to_numeric)
test = test.drop(columns=["사고일시"])

location_pattern = r"(\S+) (\S+) (\S+)"
test[["도시", "구", "동"]] = test["시군구"].str.extract(location_pattern)
test = test.drop(columns=["시군구"])
# test도 train처럼 맞춰주기

CCTV = process_location_data(CCTV, "소재지지번주소", ["소재지지번주소", "우편번호"])
light = process_location_data(light, "소재지지번주소", ["소재지지번주소", "우편번호"])
parking = process_location_data(parking, "소재지지번주소", ["소재지지번주소", "우편번호"])
child_protect = process_location_data(child_protect, "소재지지번주소", ["소재지지번주소", "우편번호"])
CCTV_notnull = CCTV.dropna(subset=["동"])
light_notnull = light.dropna(subset=["동"])
parking_notnull = parking.dropna(subset=["동"])
child_protect_notnull = child_protect.dropna(subset=["동"])

child_protect_notnull["주소"] = (
    child_protect_notnull["구"].astype(str)
    + "-"
    + child_protect_notnull["동"].astype(str)
)
child_protect_notnull.drop(["도시", "구", "동"], axis=1)

# 어린이 보호구역 개수
child_protect_df = (
    child_protect_notnull.groupby("주소").size().reset_index(name="어린이보호구역개수")
)
# 보안등 개수
light_df = light_notnull.groupby(["구", "동"]).size().reset_index(name="보안등개수")

CCTV_num = CCTV_notnull.groupby(["구", "동"]).size().reset_index(name="CCTV개수")
CCTV_pre = CCTV_notnull[["동", "단속구분"]].groupby(["동", "단속구분"]).size().unstack()
CCTV_cat = CCTV_pre.add_prefix("단속구분_")
CCTV_cat = CCTV_cat.fillna(0)
CCTV_df = pd.merge(CCTV_num, CCTV_cat, how="left", on="동")

CCTV_df = CCTV_df.groupby(["구", "동"])["단속구분_1"].unique().reset_index()


CCTV_df = CCTV_df.explode("단속구분_1").reset_index(drop=True)

# Convert the dtype of "단속구분_1" to integer
CCTV_df["단속구분_1"] = CCTV_df["단속구분_1"].astype(int)

# Display the result
CCTV_df.rename(columns={"단속구분_1": "과속단속"}, inplace=True)


CCTV_df["주소"] = CCTV_df["구"].astype(str) + "-" + CCTV_df["동"].astype(str)
CCTV_df.drop(["구", "동"], axis=1, inplace=True)
train["주소"] = train["구"].astype(str) + "-" + train["동"].astype(str)
test["주소"] = test["구"].astype(str) + "-" + test["동"].astype(str)
crossboard["주소"] = crossboard["구"].astype(str) + "-" + crossboard["법정동"].astype(str)
crossboard.drop(["구", "법정동"], axis=1, inplace=True)
numcar["주소"] = numcar["구군"].astype(str) + "-" + numcar["읍면동"].astype(str)
numcar.drop(["구군", "읍면동", "특수", "승용", "승합", "화물"], axis=1, inplace=True)
light_df["주소"] = light_df["구"].astype(str) + "-" + light_df["동"].astype(str)
light_df.drop(["구", "동"], axis=1, inplace=True)
senior_center["주소"] = (
    senior_center["구"].astype(str) + "-" + senior_center["법정동"].astype(str)
)
senior_center.drop(["구", "법정동"], axis=1, inplace=True)


road_pattern = r"(.+) - (.+)"
train[["도로형태1", "도로형태2"]] = train["도로형태"].str.extract(road_pattern)
train = train.drop(columns=["도로형태"])

test[["도로형태1", "도로형태2"]] = test["도로형태"].str.extract(road_pattern)
test = test.drop(columns=["도로형태"])

columns_to_drop = ["ID", "도시", "구", "동"]
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

train_x = train[test.columns].copy()  # test.columns 값만 넣기
train_y = train["ECLO"].copy()
test_x = test

join_1 = pd.merge(train_x, light_df, how="left", on="주소")
join_2 = pd.merge(join_1, numcar, how="left", on="주소")
join_3 = pd.merge(join_2, CCTV_df, how="left", on="주소")
join_4 = pd.merge(join_3, senior_center, how="left", on="주소")
join_5 = pd.merge(join_4, child_protect_df, how="left", on="주소")
train_x = join_5.fillna(0).copy()

join_1 = pd.merge(test_x, light_df, how="left", on="주소")
join_2 = pd.merge(join_1, numcar, how="left", on="주소")
join_3 = pd.merge(join_2, CCTV_df, how="left", on="주소")
join_4 = pd.merge(join_3, senior_center, how="left", on="주소")
join_5 = pd.merge(join_4, child_protect_df, how="left", on="주소")
test_x = join_5.fillna(0).copy()

from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder

categorical_features = list(
    train_x.dtypes[train_x.dtypes == "object"].index
)  # object값 list에 넣음


for i in categorical_features:  # 인코딩 적용한 값 반환
    le = TargetEncoder(cols=[i])
    train_x[i] = le.fit_transform(train_x[i], train_y)
    test_x[i] = le.transform(test_x[i])
# target encdoer 주의사항: train은 fit_transform인 반면, test는 transform만 진행!

# 제거할 열 이름 목록
columns_to_drop = ["연", "월"]

# train_x와 test_x에서 해당 열들을 제거
train_x = train_x.drop(columns=columns_to_drop, axis=1)
test_x = test_x.drop(columns=columns_to_drop, axis=1)

from sklearn.preprocessing import StandardScaler

# Assuming train_x and test_x are your feature matrices

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1 as l1_regularizer, l2 as l2_regularizer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from keras.wrappers.scikit_learn import KerasRegressor

# GPU 메모리 관리 설정
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # 현재 프로그램에 필요한 만큼의 GPU 메모리만 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 성장을 설정해야만 합니다
        print(e)


# RMSLE 손실 함수 정의
def rmsle(y_true, y_pred):
    y_true = tf.maximum(tf.cast(y_true, tf.float32), 0)
    y_pred = tf.maximum(tf.cast(y_pred, tf.float32), 0)
    squared_error = tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))
    return tf.sqrt(tf.reduce_mean(squared_error))


# 모델 생성 함수 정의
def create_model(learning_rate, l1_reg, l2_reg):
    input_layer = tf.keras.Input(shape=(len(train_x.columns),))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    x = tf.keras.layers.Dense(
        16, activation="relu", kernel_regularizer=l1_regularizer(l1_reg)
    )(x)
    x = tf.keras.layers.Dense(
        32, activation="relu", kernel_regularizer=l2_regularizer(l2_reg)
    )(x)
    output_layer = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate), loss=rmsle, metrics=[rmsle]
    )
    return model


def loss_fn(y_true, y_pred):
    return rmsle(y_true, y_pred)  # 차이 손실함수 반환


def metric_fn(y_true, y_pred):
    return rmsle(y_true, y_pred)


callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=3, min_lr=0.00001
    ),
    tf.keras.callbacks.TerminateOnNaN(),
]
"""
#EarlyStopping
val_loss: 데이터 손실을 관찰하겠다는 의미
patience: 검증데이터의 손실이 (30) epoch동안 개선되지 않으면 훈련 중단
verbose = 2:  조기 주단이 발생했을 때 화면에 로그를 표시
#mode = min: 관찰하고 있는 메트릭이 최소화 (손실 최소화)
restore_best_weights = true: 조기 중단 발생시 가장 좋은 가중치로 모델 복원

#ReduceLROnPlateau: 학습률을 동적으로 조정하는데 사용
monitor = val_loss: 검증 데이터이 손실 관찰
factor = 0.8: 손실이 개선되지 않을 때 학습률 80% 감소
patience: 3에포크 동안 손실 개선되지 않으면 학습률 조정
min_lr = 0.00001은 학습률의 하한선 설정ㅋ

#TreminateOnNaN:
수치적 불안정성으로 인해 손실이 NAN이 되는 경우 훈련 즉시 중단.


"""
import tensorflow as tf
from tensorflow.keras.regularizers import l1 as l1_regularizer, l2 as l2_regularizer

Best = {"batch_size": 64, "l1_reg": 0.0001, "l2_reg": 0.0001, "learning_rate": 0.01}


# 최적화된 하이퍼파라미터를 받아 모델을 생성하는 함수
def create_optimized_model(l1_reg, l2_reg, learning_rate):
    input_layer = tf.keras.Input(shape=(len(train_x.columns),))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    x = tf.keras.layers.Dense(
        16, activation="relu", kernel_regularizer=l1_regularizer(l1_reg)
    )(x)
    x = tf.keras.layers.Dense(
        32, activation="relu", kernel_regularizer=l2_regularizer(l2_reg)
    )(x)
    output_layer = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss_fn,
        metrics=[metric_fn],
    )
    return model


# 최적화된 하이퍼파라미터를 사용하여 모델 생성
best_params = Best
optimized_model = create_optimized_model(
    best_params["l1_reg"], best_params["l2_reg"], best_params["learning_rate"]
)

# 모델 학습
history = optimized_model.fit(
    train_x_scaled.astype("float32"),
    train_y.astype("float32"),
    epochs=100,
    batch_size=best_params["batch_size"],
    verbose=1,
    validation_split=0.1,
    callbacks=callbacks_list,
)
sample_submission = pd.read_csv(path + "./sample_submission.csv")

sample_submission["ECLO"] = optimized_model.predict(test_x_scaled.astype("float32"))

sample_submission.to_csv("tttt.csv", index=False)
train_y.to_csv("train_y.csv", index=False)
