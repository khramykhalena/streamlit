import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

def calculate_moving_average(df, window=30):
    df['30_day_avg'] = df['temperature'].rolling(window=window).mean()
    return df

def calculate_seasonal_stats(df):
    seasonal_stats = df.groupby(['city', 'season']).agg({'temperature': ['mean', 'std']})
    seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns.values]
    return seasonal_stats

def detect_anomalies(df, seasonal_stats):
    df = df.merge(seasonal_stats, on=['city', 'season'])
    df['anomaly'] = (df['temperature'] < (df['temperature_mean'] - 2 * df['temperature_std'])) | (df['temperature'] > (df['temperature_mean'] + 2 * df['temperature_std']))
    return df

def get_current_temp(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        return None

def main():
    st.title("Анализ температурных данных")

    uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:", df.columns)

        if 'date' in df.columns and 'temperature' in df.columns and 'city' in df.columns and 'season' in df.columns:
            df = calculate_moving_average(df)
            seasonal_stats = calculate_seasonal_stats(df)
            df = detect_anomalies(df, seasonal_stats)

            st.write("Описательная статистика по историческим данным:")
            st.write(df.describe())

            st.write("Временной ряд температур с выделением аномалий:")
            fig, ax = plt.subplots()
            ax.plot(df['date'], df['temperature'], label='Температура')
            ax.scatter(df[df['anomaly']]['date'], df[df['anomaly']]['temperature'], color='red', label='Аномалии')
            ax.legend()
            st.pyplot(fig)

            st.write("Сезонные профили с указанием среднего и стандартного отклонения:")
            st.write(seasonal_stats)

            city = st.selectbox("Выберите город", df['city'].unique())
            api_key = st.text_input("Введите API-ключ OpenWeatherMap")

            if api_key:
                current_temp = get_current_temp(api_key, city)
                if current_temp is not None:
                    st.write(f"Текущая температура в {city}: {current_temp}°C")

                    season = df[df['city'] == city]['season'].mode()[0]
                    mean_temp = seasonal_stats.loc[(city, season), 'temperature_mean']
                    std_temp = seasonal_stats.loc[(city, season), 'temperature_std']
                    is_normal = (current_temp >= (mean_temp - 2 * std_temp)) and (current_temp <= (mean_temp + 2 * std_temp))

                    if is_normal:
                        st.write("Текущая температура находится в пределах нормы.")
                    else:
                        st.write("Текущая температура аномальна.")
                else:
                    st.error("Не удалось получить данные о температуре. Проверьте API-ключ и название города.")
            else:
                st.warning("Введите API-ключ для получения текущей температуры.")
        else:
            st.error("Файл не содержит необходимых столбцов 'date', 'temperature', 'city' или 'season'.")

if __name__ == "__main__":
    main()
