import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import aiohttp
import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg, stddev
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Загрузка данных с кэшированием
@st.cache_data
def load():
    d = pd.read_csv('temperature_data.csv')
    d['ts'] = pd.to_datetime(d['timestamp'])
    return d

# Распараллеливание вычислений с использованием PySpark
def calculate_rolling_stats_spark(df):
    spark = SparkSession.builder.appName("TemperatureAnalysis").getOrCreate()
    spark_df = spark.createDataFrame(df)
    window = Window.partitionBy("city").orderBy("ts").rowsBetween(-29, 0)
    spark_df = spark_df.withColumn("rm", avg(col("temperature")).over(window))
    spark_df = spark_df.withColumn("rs", stddev(col("temperature")).over(window))
    return spark_df.toPandas()

# Получение текущей температуры синхронно
def get_temp_sync(k, c):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={c}&appid={k}&units=metric"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()['main']['temp']
    else:
        err = r.json().get('message', 'Ошибка')
        if r.status_code == 401:
            st.error(f"Ошибка: {err}. Проверьте API ключ.")
        else:
            st.error(f"Ошибка при запросе к API: {r.status_code} - {err}")
        return None

# Получение текущей температуры асинхронно
async def get_temp_async(k, c):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={c}&appid={k}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            if r.status == 200:
                data = await r.json()
                return data['main']['temp']
            else:
                err = await r.json()
                st.error(f"Ошибка: {err.get('message', 'Неизвестная ошибка')}")
                return None

# Построение долгосрочных трендов
def plot_long_term_trend(df, city):
    city_df = df[df['city'] == city]
    X = np.arange(len(city_df)).reshape(-1, 1)
    y = city_df['temperature'].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    
    fig = px.line(city_df, x='ts', y='temperature', title=f"Долгосрочный тренд температуры в {city}")
    fig.add_scatter(x=city_df['ts'], y=trend, mode='lines', name='Тренд', line=dict(color='red'))
    st.plotly_chart(fig)

def main():
    st.title("Анализ температурных данных")

    d = load()

    # Распараллеливание вычислений
    d = calculate_rolling_stats_spark(d)
    d['a'] = (d['temperature'] < (d['rm'] - 2 * d['rs'])) | (d['temperature'] > (d['rm'] + 2 * d['rs']))

    # Загрузка пользовательского файла
    f = st.file_uploader("Загрузите файл с историческими данными", type="csv")
    if f is not None:
        d = pd.read_csv(f)
        d['ts'] = pd.to_datetime(d['timestamp'])
        d = calculate_rolling_stats_spark(d)
        d['a'] = (d['temperature'] < (d['rm'] - 2 * d['rs'])) | (d['temperature'] > (d['rm'] + 2 * d['rs']))

    # Выбор города
    c = st.selectbox("Выберите город", d['city'].unique())

    # Ввод API ключа
    k = st.text_input("Введите API ключ OpenWeatherMap")

    # Получение текущей температуры
    if k:
        if st.button("Получить температуру синхронно"):
            t = get_temp_sync(k, c)
            if t is not None:
                st.write(f"Текущая температура в {c}: {t}°C")
                s = d[(d['city'] == c) & (d['season'] == d[d['city'] == c]['season'].iloc[-1])]
                m = s['temperature'].mean()
                s_std = s['temperature'].std()
                if (t < m - 2 * s_std) or (t > m + 2 * s_std):
                    st.write("Текущая температура является аномальной.")
                else:
                    st.write("Текущая температура в пределах нормы.")

        if st.button("Получить температуру асинхронно"):
            t = asyncio.run(get_temp_async(k, c))
            if t is not None:
                st.write(f"Текущая температура в {c}: {t}°C")
                s = d[(d['city'] == c) & (d['season'] == d[d['city'] == c]['season'].iloc[-1])]
                m = s['temperature'].mean()
                s_std = s['temperature'].std()
                if (t < m - 2 * s_std) or (t > m + 2 * s_std):
                    st.write("Текущая температура является аномальной.")
                else:
                    st.write("Текущая температура в пределах нормы.")
    else:
        st.write("Введите API ключ для получения текущей температуры.")

    # Временной ряд температур
    if st.checkbox("Показать временной ряд температур"):
        cd = d[d['city'] == c]
        fig = px.line(cd, x='ts', y='temperature', title=f"Температура в {c}")
        fig.add_scatter(x=cd[cd['a']]['ts'], y=cd[cd['a']]['temperature'], mode='markers', name='Аномалии', marker=dict(color='red'))
        st.plotly_chart(fig)

    # Сезонные профили
    if st.checkbox("Показать сезонные профили"):
        sd = d[d['city'] == c]
        m = sd.groupby('season')['temperature'].mean()
        s_std = sd.groupby('season')['temperature'].std()
        fig = px.bar(x=m.index, y=m, error_y=s_std, title=f"Сезонные профили температуры в {c}")
        st.plotly_chart(fig)

    # Долгосрочные тренды
    if st.checkbox("Показать долгосрочные тренды"):
        plot_long_term_trend(d, c)

if __name__ == "__main__":
    main()
