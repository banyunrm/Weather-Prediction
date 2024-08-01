
import streamlit as st
import pandas as pd
import pickle

# Memuat kembali model yang telah disimpan
with open('best_model_rf.pkl', 'rb') as file:
    pipeline_rf = pickle.load(file)

def run():
    st.title('Prediction of Weather Conditions')

    with st.form('form_weather'):
        temp = st.slider('Temperature', min_value=0, max_value=50, value=25, help='Slide temperature value')
        humidity = st.slider('Humidity', min_value=0, max_value=100, value=50, help='Slide humidity value')
        wind_speed = st.slider('Wind Speed', min_value=0, max_value=100, value=10, help='Slide wind speed value')
        precipitation_pct = st.slider('Precipitation Percentage', min_value=0, max_value=100, value=20, help='Slide precipitation percentage')
        atm_pressure = st.slider('Atmospheric Pressure', min_value=950, max_value=1050, value=1000, help='Slide atmospheric pressure value')
        uv_index = st.slider('UV Index', min_value=0, max_value=11, value=5, help='Slide UV index value')
        visibility_km = st.slider('Visibility (km)', min_value=0, max_value=20, value=10, help='Slide visibility in kilometers')
        cloud_cover = st.selectbox('Cloud Cover', options=['Clear', 'Partly Cloudy', 'Cloudy', 'Overcast'])
        season = st.selectbox('Season', options=['Spring', 'Summer', 'Autumn', 'Winter'])
        location = st.selectbox('Location', options=['Urban', 'Suburban', 'Rural'])

        # Tombol kirim
        submitted = st.form_submit_button('Predict')

    data_inf = {
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation_pct": precipitation_pct,
        "atm_pressure": atm_pressure,
        "uv_index": uv_index,
        "visibility_km": visibility_km,
        "cloud_cover": cloud_cover,
        "season": season,
        "location": location
    }

    data_inf = pd.DataFrame([data_inf])

    if submitted:
        # Memprediksi menggunakan model terbaik yang telah dimuat
        y_pred = pipeline_rf.predict(data_inf)
        st.write('### Predicted Weather Condition: ', str(y_pred[0]))

if __name__ == "__main__":
    run()
