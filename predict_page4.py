import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
import streamlit as st
import json
from PIL import Image
import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import normalize, scale
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from annotated_text import annotated_text



warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


audio_file = open('ABBA-MONEYMONEYMONEY.mp3', 'rb')
audio_bytes = audio_file.read()

def load_model():
    with open('saved_steps (1).pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

final_model = data["model"]
# df_new = data["dataframe"]
scaler = data["scaler"]

########################################
# Lottie Functions
########################################


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

########################################
########################################


def generate_df_new(df):
    g = ["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer", "Analytics Engineer",
         "Data Architect"]
    new = [col for col in df.index if df.job_title[col] in g]

    df = df.loc[new,]
    df = df.reset_index()
    del df["index"]

    g = ["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer", "Analytics Engineer",
         "Data Architect"]
    new = [col for col in df.index if df.job_title[col] in g]

    df = df.loc[new,]
    df = df.reset_index()
    del df["index"]

    df.company_size = df.company_size.replace({'S': 1, 'M': 2, 'L': 3})
    df['remote_ratio'] = df['remote_ratio'].replace({100: 3, 50: 2, 0: 1})
    df['experience_level'] = df['experience_level'].replace({'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4})
    df['employment_type'] = df['employment_type'].replace({'FL': 1, 'PT': 2, 'FT': 3, 'CT': 4, })

    x = df
    x = x.astype(str)
    x["cluster"] = ['_'.join(i) for i in x.drop(["salary", "salary_in_usd"], axis=1).values]
    x = x['cluster']
    df["cluster"] = x
    df_new = df
    df_new['country_median_salary'] = df_new.groupby(
        ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size'])['salary'].transform('median')
    df_new['country_mean_salary'] = df_new.groupby(
        ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size'])['salary'].transform('mean')
    df_new['country_min_salary'] = df_new.groupby(
        ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size'])['salary'].transform('min')
    df_new['country_max_salary'] = df_new.groupby(
        ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size'])['salary'].transform('max')

    a = []
    for (i, j) in zip(df_new['employee_residence'], df_new['company_location']):
        if i == j:
            a.append('Yes')
        else:
            a.append('No')

    df_new['work_own_country'] = a

    return df_new

#
# def regression_(x_train, x_test, y_train, y_test):
#     lr = LinearRegression()
#     rf = RandomForestRegressor()
#     lg = LGBMRegressor()
#     # ct = CatBoostRegressor()
#     r = Ridge()
#     l = Lasso()
#     e = ElasticNet()
#     kn = KNeighborsRegressor()
#     et = ExtraTreeRegressor()
#     gb = GradientBoostingRegressor()
#     dt = DecisionTreeRegressor()
#     xgb = XGBRegressor()
#
#     algos = [lr, rf, lg, r, l, e, kn, et, gb, dt, xgb]
#     algos_names = ['LinearRegressor', "rf", "lgbm", 'Ridge', 'Lasso', 'ElasticNet', 'KNeighbors', 'ExtraTree',
#                    'GradientBoosting',
#                    'DecisionTree', 'XGB']
#
#     r_score = []
#     mse = []
#     mae = []
#
#     result = pd.DataFrame(columns=['R_square', 'MSE', 'MAE'], index=algos_names)
#
#     for algo in algos:
#         pred = algo.fit(x_train, y_train).predict(x_test)
#         r_score.append(r2_score(y_test, pred))
#         mse.append(mean_squared_error(y_test, pred) ** .5)
#         mae.append(mean_absolute_error(y_test, pred))
#
#     result.R_square = r_score
#     result.MSE = mse
#     result.MAE = mae
#
#     return result.sort_values('R_square', ascending=False)


def show_predict_page():
    df = pd.read_csv("ds_salaries.csv")

    df_new = generate_df_new(df)


    st.title("Data Scientist Salary Prediction")

    st.write("""### We need some information to predict the salary""")


    job_titles = (
        "Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
        "Analytics Engineer", "Data Architect",
    )

    work_years = (
        2020,
        2021,
        2022,
        2023,
    )

    #'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4
    experience_levels = (
#        1, 2, 3, 4,
        "Entry Level", "Middle/Intermediate Level", "Senior Level", "Executive Level",
    )

    #'FL':1 ,'PT': 2, 'FT': 3, 'CT': 4
    employment_types = (
        "Freelancer", "Part Time", "Full Time", "Contractor",
    )
    employee_residences = (
        "ES", "US", "CA", "DE", "GB", "NG", "IN", "HK", "PT", "NL", "ES", "CH", "CF", "FR", "AU", "FI", "UA", "IE",
        "IL", "GH", "AT", "CO", "SG", "SE", "SI", "MX", "UZ", "BR", "TH", "HR", "PL", "KW", "VN", "CY", "AR", "AM",
        "BA", "KE", "GR", "MK", "LV", "RO", "PK", "IT", "MA", "LT", "BE", "AS", "IR", "HU", "SK", "CN", "CZ", "CR",
        "TR", "CL", "PR", "DK", "BO", "PH", "DO", "EG", "ID", "AE", "MY", "JP", "EE", "HN", "TN", "RU", "DZ", "IQ",
        "BG", "JE", "RS", "NZ", "MD", "LU", "MT",
    )

    #100: 3, 50: 2, 0: 1
    remote_ratios = (
        "Onsite", "Hybrid", "Full Remote",
    )
    company_locations = (
        "ES", "US", "CA", "DE", "GB", "NG", "IN", "HK", "NL", "ES", "CH", "CF", "FR", "FI", "UA", "IE", "IL", "GH",
        "CO", "SG", "AU", "SE", "SI", "MX", "BR", "PT", "RU", "TH", "HR", "VN", "EE", "AM", "BA", "KE", "GR", "MK",
        "LV", "RO", "PK", "IT", "MA", "PL", "AL", "AR", "LT", "AS", "CR", "IR", "BS", "HU", "AT", "SK", "CZ", "TR",
        "PR", "DK", "BO", "PH", "BE", "ID", "EG", "AE", "LU", "MY", "HN", "JP", "DZ", "IQ", "CN", "NZ", "CL", "MD",
        "MT",
    )

    #'S': 1, 'M': 2, 'L': 3
    company_sizes = (
#        1, 2, 3,
        "Small", "Medium", "Large",
    )
    ############################

    job_title = st.selectbox("Job Title", job_titles)
    work_year = st.slider("Work Year", min_value=2020,max_value=2022,value=2020,step=1)
    experience_level = st.radio("Experience Level",experience_levels, index=1)
    employment_type = st.selectbox("Employment Type", employment_types)
    employee_residence = st.selectbox("Employee Residence", employee_residences)
    remote_ratio = st.radio("Remote Ratio", remote_ratios, index=1)
    company_location = st.selectbox("Company Location", company_locations)
    company_size = st.selectbox("Company Size", company_sizes)
    print(type(company_size))
    #########################################################
    print(company_size)
    #######################################################
    ok = st.button("Calculate Salary")
    #########################################################

    # 'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4
    if experience_level == "Entry Level":
        experience_level = 1
    elif experience_level == "Middle/Intermediate Level":
        experience_level = 2
    elif experience_level == "Senior Level":
        experience_level = 3
    elif experience_level == "Executive Level":
        experience_level = 4

    # 'FL':1 ,'PT': 2, 'FT': 3, 'CT': 4
    if employment_type == "Freelancer":
        employment_type = 1
    elif employment_type == "Part Time":
        employment_type = 2
    elif employment_type == "Full Time":
        employment_type = 3
    elif employment_type == "Contractor":
        employment_type = 4

    # 'S': 1, 'M': 2, 'L': 3
    if company_size == "Small":
        company_size = 1
    elif company_size == "Medium":
        company_size = 2
    elif company_size == "Large":
        company_size = 3
    print(company_size)

    #100: 3, 50: 2, 0: 1
    if remote_ratio == "Full Remote":
        remote_ratio = 3
    elif remote_ratio == "Hybrid":
        remote_ratio = 2
    elif remote_ratio == "Onsite":
        remote_ratio = 1

    if ok:

        data = {'work_year':[work_year], 'experience_level':[experience_level], 'employment_type':[employment_type],
                'job_title':[job_title], 'employee_residence':[employee_residence], 'remote_ratio':[remote_ratio],
                'company_location':[company_location], 'company_size':[company_size]
                }
        X = pd.DataFrame(data)
        # X = np.array([[work_year, experience_level, employment_type, job_title, employee_residence, remote_ratio,
        #                 company_location, company_size]])

        x = X
        x = x.astype(str)
        x["cluster"] = ['_'.join(i) for i in x.values]
        x = x['cluster']
        X["cluster"] = x
        X_new = X

        X_new['country_median_salary'] = df_new['country_median_salary']
        X_new['country_mean_salary'] = df_new.groupby(
            ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
             'company_location', 'company_size'])['salary'].transform('mean')
        X_new['country_min_salary'] = df_new['country_min_salary']
        X_new['country_max_salary'] = df_new['country_max_salary']

        b = []
        for (i, j) in zip(X_new['employee_residence'], X_new['company_location']):
            if i == j:
                b.append('Yes')
            else:
                b.append('No')

        X_new['work_own_country'] = b

        X_new["salary"] = 0
        X_new['salary_in_usd'] = 0
        X_new['salary_currency'] = "USD"

        df_temp = df_new.drop(2984)

        df_final = pd.concat([df_temp, X_new], axis=0)

        df_final = df_final.reset_index()
        df_final = df_final.drop(columns=["index"])

        cats = df_final.select_dtypes(include="object").columns

        def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
            dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
            return dataframe

        X2 = one_hot_encoder(df_final, cats)

        X2 = X2.drop(['salary', 'salary_in_usd','country_mean_salary'], axis=1)

        X2 = scaler.fit_transform(X2)

        Sub_pred = final_model.predict(X2)
        Sub_pred = Sub_pred.reshape(-1, 1)
        Sub_pred = np.exp(Sub_pred)
        salary = Sub_pred[2984][0]


        st.subheader(f"The estimated salary is ${salary:.2f}")

        if salary > 120000:
            st.subheader(f"WOOWWWWWWWW!! FANTASTIC")
            st.audio(audio_bytes, format='audio/mp3',start_time=46)

            lottie_coding = load_lottiefile("79808-green-money-falling.json")  # replace link to local lottie file
            lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

            st_lottie(
                lottie_coding,
                speed=0.5,
                reverse=False,
                loop=True,
                quality="low",  # medium ; high
            #    renderer="svg",  # canvas
                height=None,
                width=None,
                key=None,
            )

        elif salary <80000:
            st.subheader(f"Garibanın yüzü gülür mü :((((")
            audio_file2 = open('kucuk-emrah-yarali.mp3', 'rb')
            audio_bytes2 = audio_file2.read()
            st.audio(audio_bytes2, format='audio/mp3', start_time=80)
            image = Image.open('Acıların_Cocugu.jpg')

            st.image(image, width=400)