import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.title("Дашборд по демографическим показателям стран")
st.markdown("""Данное приложение содержит реализацию следующих функций:
- Диаграмма распределения населения по годам (реализация через производную)
- Относительный прирост по годам
- Соотношение мужчин/женщин по годам
- Возрастно-половая пирамида по годам
- Прогноз на 10/25/50 лет, основанный на наборе данных""")

@st.cache_data()
def load_data():
    table = pd.read_csv("population.csv")
    table.dropna(subset=["Series Name", "Country Name"], inplace=True)
    indexes = table[table["Series Name"].str.contains("65 and")].index
    table.drop(indexes, inplace=True)
    return table

table = load_data()
years = table.columns[2:].tolist()
countries = table.groupby("Country Name").sum()

st.write(table.head())

st.header("Выберите страну")
country = st.selectbox("Страна", table["Country Name"].unique())

if country:
    total_population = countries.loc[country]
    years_int = [int(year) for year in years]

    st.subheader(f"Диаграмма распределения населения по годам")
    diff = []
    for i, year in enumerate(years):
        if i > 0:
            prev_year = years[i - 1]
            delta_x = 1
            delta_y = total_population.loc[year] - total_population.loc[prev_year]
            diff.append(delta_y / delta_x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_int[1:], diff)
    ax.grid(True)
    ax.axhline(y=0, color='darkred', linestyle='--')
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Относительный прирост по годам")
    growth = []
    for i, year in enumerate(years):
        if i > 0:
            prev_year = years[i - 1]
            base = total_population.loc[prev_year]
            delta_y = total_population.loc[year] - total_population.loc[prev_year]
            growth.append(delta_y/base)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_int[1:], growth)
    ax.grid(True)
    ax.axhline(y=0, color='darkred', linestyle='--')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Соотношение мужчин/женщин по годам")
    males = []
    females = []
    relation = []
    for i, year in enumerate(years):
        male_population = table[(table["Country Name"]==country) &
                               (table["Series Name"].str.contains(" male"))].sum()[year]
        female_population = table[(table["Country Name"]==country) &
                                 (table["Series Name"].str.contains("female"))].sum()[year]
        males.append(male_population)
        females.append(female_population)
        relation.append(male_population/female_population)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_int, males, color="darkblue", label="Мужчины")
    ax.plot(years_int, females, color="darkred", label="Женщины")
    ax.set_title("График количества мужчин и женщин в популяции по годам")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_int, relation)
    ax.axhline(y=1, color="darkred", linestyle='--', label='Равенство (1:1)')
    ax.set_title("График отношения количества мужчин и женщин по годам")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Возрастно-половая пирамида по годам")
    year_selected = st.selectbox("Выберите год", years)

    if year_selected:
        male_ages_df = table[(table["Country Name"] == country) &
                           (table["Series Name"].str.contains(" male"))].loc[:,["Series Name", year_selected]]
        female_ages_df = table[(table["Country Name"] == country) &
                           (table["Series Name"].str.contains("female"))].loc[:,["Series Name", year_selected]]
        male_ages = []
        female_ages = []
        ages = []
        for i in range(len(male_ages_df["Series Name"])):
            if i < 2:
                age = '0' + str(5*i) + '-0' + str(5*i + 4)
                ages.append(age)
            elif i < 16:
                age = str(5*i) + '-' + str(5*i + 4)
                ages.append(age)
            else:
                age = str(5*i)
                ages.append(age+'+')
            male_ages.append(-1*(male_ages_df[male_ages_df["Series Name"].str.contains(age)].iloc[0,1]))
            female_ages.append(female_ages_df[female_ages_df["Series Name"].str.contains(age)].iloc[0,1])
            min_ages = []
            min_ages_neg = []
            for i in range(len(ages)):
                min_ages.append(min(abs(male_ages[i]), female_ages[i]))
                min_ages_neg.append(-min(abs(male_ages[i]), female_ages[i]))
        df = pd.DataFrame({"male": male_ages, "female": female_ages,
                           "age": ages, "min_ages": min_ages, "min_ages_neg": min_ages_neg})
        age_class = ages[::-1]
        fig, ax = plt.subplots()
        sns.barplot(ax=ax, data=df, x="male", y="age", order=age_class, lw=0, color="Darkblue", label="Навес мужчин")
        sns.barplot(ax=ax, data=df, x="female", y="age", order=age_class, lw=0, color="Darkred", label="Навес женщин")
        sns.barplot(ax=ax, data=df, x="min_ages", y="age", order=age_class, lw=0, color="Red", label="Женщины")
        sns.barplot(ax=ax, data=df, x="min_ages_neg", y="age", order=age_class, lw=0, color="SteelBlue", label="Мужчины")
        ax.set(xlabel="Население", ylabel="Возрастная группа",
               title=f"Возрастно-половая пирамида за {year_selected} год")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Прогноз на 10/25/50 лет, основанный на наборе данных")
    period = st.selectbox("Выберите период", [10, 25, 50])

    if period:
        def regression(period):
            data = np.log(np.array(total_population[years].values, dtype=float))
            shaped_years = np.array(years_int).reshape(-1, 1)
            model = LinearRegression()
            model.fit(shaped_years, data)
            log_population = model.predict(shaped_years)
            population = np.exp(log_population)
            period_years = np.arange(years_int[-1]+1, years_int[-1]+period+1)
            log_predict = model.predict(period_years.reshape(-1, 1))
            prediction = np.exp(log_predict)

            fig, ax = plt.subplots()
            ax.plot(np.concatenate((years_int, period_years)), np.concatenate((population, prediction)),
                    label=f"Предсказанная численность населения на ближайшие {period} лет")
            ax.plot(years_int, total_population[years].values, label="Реальные данные")
            ax.set_title("Графики общей численности населения")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        regression(period)