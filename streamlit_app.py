import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv('temperature_data.csv')
    return data

def get_current_temperature(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        return None

def main():
    st.title("Анализ температурных данных")

    data = load_data()

    uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    city = st.selectbox("Выберите город", data['city'].unique())

    api_key = st.text_input("Введите API ключ OpenWeatherMap")

    if api_key:
        current_temp = get_current_temperature(api_key, city)
        if current_temp is not None:
            st.write(f"Текущая температура в {city}: {current_temp}°C")

            season_data = data[(data['city'] == city) & (data['season'] == data[data['city'] == city]['season'].iloc[-1])]
            mean_temp = season_data['temperature'].mean()
            std_temp = season_data['temperature'].std()

            if (current_temp < mean_temp - 2 * std_temp) or (current_temp > mean_temp + 2 * std_temp):
                st.write("Текущая температура является аномальной.")
            else:
                st.write("Текущая температура в пределах нормы.")
        else:
            st.write("Не удалось получить текущую температуру. Проверьте API ключ и название города.")
    else:
        st.write("Введите API ключ для получения текущей температуры.")

    if st.checkbox("Показать временной ряд температур"):
        city_data = data[data['city'] == city]
        plt.figure(figsize=(10, 5))
        plt.plot(city_data['timestamp'], city_data['temperature'], label='Температура')
        plt.plot(city_data['timestamp'], city_data['rolling_mean'], label='Скользящее среднее')
        plt.scatter(city_data[city_data['anomaly']]['timestamp'], city_data[city_data['anomaly']]['temperature'], color='red', label='Аномалии')
        plt.legend()
        st.pyplot(plt)

    if st.checkbox("Показать сезонные профили"):
        season_data = data[data['city'] == city]
        season_mean = season_data.groupby('season')['temperature'].mean()
        season_std = season_data.groupby('season')['temperature'].std()
        plt.figure(figsize=(10, 5))
        plt.bar(season_mean.index, season_mean, yerr=season_std, capsize=5)
        plt.title(f"Сезонные профили температуры в {city}")
        st.pyplot(plt)

if __name__ == "__main__":
    main()