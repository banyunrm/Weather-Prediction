import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def run():
    # Membuat judul
    st.title('Prediction of Weather Conditions')

    # Menampilkan gambar
    image = Image.open('weather.jpeg')
    st.image(image, caption='Predict the weather conditions using machine learning models')

    st.markdown('------')

    # Sub judul untuk Exploratory Data Analysis
    st.subheader('Exploratory Data Analysis Weather Conditions')

    # Memanggil dataset
    credit_card_data = pd.read_csv('weather_classification_data.csv')
    st.write(credit_card_data)

    # Rata-rata Suhu Berdasarkan Musim
    st.write("#### Average Temperature by Season")
    image = Image.open('rata-rata_suhu_berdasarkan_musim.png')
    st.image(image, caption='Graph showing the average temperatures across different seasons')


    # Rata-rata Suhu per Lokasi
    st.write("#### Average Temperature per Site")
    image = Image.open('rata-rata_suhu_per_lokasi.png')
    st.image(image, caption='Graph depicting the average temperatures for various locations')


    # Jumlah Jenis Cuaca per Musim
    st.write("#### Number of Weather Types per Season")
    image = Image.open('jumlah_jenis_cuaca_per_musim.png')
    st.image(image, caption='Bar chart showing the number of different weather types in each season')
    
    # Rata-rata Kelembaban per Jenis Cuaca
    st.write("#### Average Humidity per Weather Type")
    image = Image.open('rata-rata_kelembaban_per_jenis_cuaca.png')
    st.image(image, caption='Graph displaying the average humidity levels for each weather type')

if __name__ == "__main__":
    run()
