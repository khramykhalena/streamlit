import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from multiprocessing import Pool
import time

@st.cache_data
def load_data():
    data = pd.read_csv('temperature_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def calculate_rolling_stats(data):
    data['rolling_mean'] = data.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).mean())
    data['rolling_std'] = data.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).std())
    data['anomaly'] = (data['temperature'] < (data['rolling_mean'] - 2 * data['rolling_std'])) | (data['temperature'] > (data['rolling_mean'] + 2 * data['rolling_std']))
    return data

def get_current_temperature(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        error_message = response.json().get('message', 'Неизвестная ошибка')
        if response.status_code == 401:
            st.error(f"Ошибка: {error_message}. Проверьте API ключ.")
        else:
            st.error(f"Ошибка при запросе к API: {response.status_code} - {error_message}")
        return None

async def get_current_temperature_async(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                error_message = (await response.json()).get('message', 'Неизвестная ошибка')
                st.error(f"Ошибка при запросе к API: {response.status} - {error_message}")
                return None

def main():
    st.set_page_config(page_title="Анализ температурных данных", layout="wide")
    st.title("🌡️ Анализ температурных данных")

    data = load_data()

    uploaded_file = st.file_uploader("Загрузите файл в формате csv", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    data = calculate_rolling_stats(data)

    city = st.selectbox("Выберите город", data['city'].unique())

    api_key = st.text_input("Введите API ключ OpenWeatherMap")

    if api_key:
        if st.button("Получить текущую температуру (синхронно)"):
            start_time = time.time()
            current_temp = get_current_temperature(api_key, city)
            end_time = time.time()
            st.write(f"Время выполнения синхронного запроса: {end_time - start_time:.2f} секунд")
            if current_temp is not None:
                st.write(f"Текущая температура в {city}: {current_temp}°C")

                season_data = data[(data['city'] == city) & (data['season'] == data[data['city'] == city]['season'].iloc[-1])]
                mean_temp = season_data['temperature'].mean()
                std_temp = season_data['temperature'].std()

                if (current_temp < mean_temp - 2 * std_temp) or (current_temp > mean_temp + 2 * std_temp):
                    st.error("Текущая температура является аномальной.")
                else:
                    st.success("Текущая температура в пределах нормы.")

    if st.checkbox("Показать временной ряд температур"):
        city_data = data[data['city'] == city]
        fig = px.line(city_data, x='timestamp', y='temperature', title=f"Температура в {city}")
        fig.add_scatter(x=city_data['timestamp'], y=city_data['rolling_mean'], mode='lines', name='Скользящее среднее')
        fig.add_scatter(x=city_data[city_data['anomaly']]['timestamp'], y=city_data[city_data['anomaly']]['temperature'], mode='markers', name='Аномалии', marker=dict(color='red'))
        st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Показать сезонные профили"):
        season_data = data[data['city'] == city]
        season_mean = season_data.groupby('season')['temperature'].mean().reset_index()
        season_std = season_data.groupby('season')['temperature'].std().reset_index()

        fig = px.bar(season_mean, x='season', y='temperature', error_y=season_std['temperature'], title=f"Сезонные профили температуры в {city}")
        st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Показать описательную статистику"):
        city_data = data[data['city'] == city]
        st.write(city_data['temperature'].describe())

if __name__ == "__main__":
    main()
