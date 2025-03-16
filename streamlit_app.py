import streamlit as st
import pandas as pd
import numpy as np
import requests
import aiohttp
import asyncio
from sklearn.linear_model import LinearRegression
import plotly.express as px

@st.cache_data
def load():
    df = pd.read_csv('temperature_data.csv')
    df['ts'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_stats(df):
    df['rm'] = df.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).mean())
    df['rs'] = df.groupby('city')['temperature'].transform(lambda x: x.rolling(window=30).std())
    df['a'] = (df['temperature'] < (df['rm'] - 2 * df['rs'])) | (df['temperature'] > (df['rm'] + 2 * df['rs']))
    return df

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

def long_term_trend(df, city):
    cdf = df[df['city'] == city]
    X = np.arange(len(cdf)).reshape(-1, 1)
    y = cdf['temperature'].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    fig = px.line(cdf, x='ts', y='temperature', title=f"Тренд температуры в {city}", labels={'ts': 'Дата', 'temperature': 'Температура (°C)'})
    fig.add_scatter(x=cdf['ts'], y=trend, mode='lines', name='Тренд', line=dict(color='red'))
    st.plotly_chart(fig)

def main():
    st.title("Анализ температурных данных")
    df = load()
    df = calculate_stats(df)
    f = st.file_uploader("Загрузите файл с данными", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        df['ts'] = pd.to_datetime(df['timestamp'])
        df = calculate_stats(df)
    c = st.selectbox("Выберите город", df['city'].unique())
    k = st.text_input("Введите API ключ OpenWeatherMap")
    if k:
        if st.button("Получить температуру синхронно"):
            t = get_temp_sync(k, c)
            if t is not None:
                st.write(f"Текущая температура в {c}: {t}°C")
                s = df[(df['city'] == c) & (df['season'] == df[df['city'] == c]['season'].iloc[-1])]
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
                s = df[(df['city'] == c) & (df['season'] == df[df['city'] == c]['season'].iloc[-1])]
                m = s['temperature'].mean()
                s_std = s['temperature'].std()
                if (t < m - 2 * s_std) or (t > m + 2 * s_std):
                    st.write("Текущая температура является аномальной.")
                else:
                    st.write("Текущая температура в пределах нормы.")
    else:
        st.write("Введите API ключ для получения текущей температуры.")
    if st.checkbox("Показать описательную статистику"):
        st.write(df[df['city'] == c]['temperature'].describe())
    if st.checkbox("Показать временной ряд температур"):
        cdf = df[df['city'] == c]
        fig = px.line(cdf, x='ts', y='temperature', title=f"Температура в {c}", labels={'ts': 'Дата', 'temperature': 'Температура (°C)'})
        fig.add_scatter(x=cdf[cdf['a']]['ts'], y=cdf[cdf['a']]['temperature'], mode='markers', name='Аномалии', marker=dict(color='red'))
        st.plotly_chart(fig)
    if st.checkbox("Показать сезонные профили"):
        sdf = df[df['city'] == c]
        m = sdf.groupby('season')['temperature'].mean()
        s_std = sdf.groupby('season')['temperature'].std()
        season_data = pd.DataFrame({
            'Сезон': m.index,
            'Средняя температура': m.values,
            'Стандартное отклонение': s_std.values
        })
        st.write("Сезонные профили температуры:")
        st.dataframe(season_data)
        fig = px.line(season_data, x='Сезон', y='Средняя температура', error_y='Стандартное отклонение', 
                      title=f"Сезонные профили температуры в {c}", labels={'Сезон': 'Сезон', 'Средняя температура': 'Температура (°C)'})
        st.plotly_chart(fig)
    if st.checkbox("Показать долгосрочные тренды"):
        long_term_trend(df, c)

if __name__ == "__main__":
    main()
