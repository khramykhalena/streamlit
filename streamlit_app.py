import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

@st.cache_data
def load():
    d = pd.read_csv('temperature_data.csv')
    d['ts'] = pd.to_datetime(d['timestamp'])
    d['rm'] = d.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).mean())
    d['rs'] = d.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).std())
    d['a'] = (d['temperature'] < (d['rm'] - 2 * d['rs'])) | (d['temperature'] > (d['rm'] + 2 * d['rs']))
    return d

def get_temp(k, c):
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

def main():
    st.title("Анализ температурных данных")
    d = load()
    f = st.file_uploader("Загрузите файл с историческими данными", type="csv")
    if f is not None:
        d = pd.read_csv(f)
        d['ts'] = pd.to_datetime(d['timestamp'])
        d['rm'] = d.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).mean())
        d['rs'] = d.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).std())
        d['a'] = (d['temperature'] < (d['rm'] - 2 * d['rs'])) | (d['temperature'] > (d['rm'] + 2 * d['rs']))

    c = st.selectbox("Выберите город", d['city'].unique())
    k = st.text_input("Введите API ключ OpenWeatherMap")
    if k:
        t = get_temp(k, c)
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
    if st.checkbox("Показать временной ряд температур"):
        cd = d[d['city'] == c]
        plt.figure(figsize=(10, 5))
        plt.plot(cd['ts'], cd['temperature'], label='Температура', color='blue')
        if 'rm' in cd.columns:
            plt.plot(cd['ts'], cd['rm'], label='Скользящее среднее', color='green') 
        else:
            st.error("Столбец 'rm' отсутствует в данных.")
        if 'a' in cd.columns:
            plt.scatter(cd[cd['a']]['ts'], cd[cd['a']]['temperature'], color='orange', label='Аномалии')
        plt.legend()
        st.pyplot(plt)
    if st.checkbox("Показать сезонные профили"):
        sd = d[d['city'] == c]
        m = sd.groupby('season')['temperature'].mean()
        s_std = sd.groupby('season')['temperature'].std()
        plt.figure(figsize=(10, 5))
        plt.plot(m.index, m, marker='o', label='Средняя температура', color='purple')
        plt.fill_between(m.index, m - 2 * s_std, m + 2 * s_std, alpha=0.2, label='±2σ', color='lightblue') 
        plt.title(f"Сезонные профили температуры в {c}")
        plt.xlabel("Сезон")
        plt.ylabel("Температура (°C)")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
