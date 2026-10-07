import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# 1. Конфігурація сторінки Streamlit
st.set_page_config(
    page_title="NY Real Estate AI Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Довідник типових географічних координат та округів Нью-Йорка
DISTRICT_DATA = {
    "Manhattan": {
        "locality": "New York County",
        "lat": 40.7831,
        "lon": -73.9712,
        "desc": "Центральний діловий та культурний центр Нью-Йорка."
    },
    "Brooklyn": {
        "locality": "Kings County",
        "lat": 40.6782,
        "lon": -73.9442,
        "desc": "Найбільш густонаселений округ із високим попитом на житло."
    },
    "Queens": {
        "locality": "Queens County",
        "lat": 40.7282,
        "lon": -73.7949,
        "desc": "Найбільший за площею округ із різноманітною забудовою."
    },
    "The Bronx": {
        "locality": "Bronx County",
        "lat": 40.8448,
        "lon": -73.8648,
        "desc": "Північний округ із доступнішим сегментом нерухомості."
    },
    "Staten Island": {
        "locality": "Richmond County",
        "lat": 40.5795,
        "lon": -74.1502,
        "desc": "Острівний округ із переважанням приватних садиб і котеджів."
    },
    "Flushing": {
        "locality": "Queens County",
        "lat": 40.7674,
        "lon": -73.8331,
        "desc": "Великий комерційно-житловий вузол у Квінзі."
    },
    "Riverdale": {
        "locality": "Bronx County",
        "lat": 40.8906,
        "lon": -73.9126,
        "desc": "Елітний зелений мікрорайон на північному заході Бронкса."
    },
    "Coney Island": {
        "locality": "Kings County",
        "lat": 40.5755,
        "lon": -73.9707,
        "desc": "Прибережний рекреаційний район на півдні Брукліна."
    },
    "Other": {
        "locality": "Other",
        "lat": 40.7128,
        "lon": -74.0060,
        "desc": "Інші прилеглі зони агломерації Нью-Йорка."
    }
}

# 3. Кешоване завантаження моделі та препроцесора
@st.cache_resource
def load_artifacts():
    """
    Завантажує натреновану Keras-модель та препроцесор Scikit-Learn.
    Здійснює пошук за різними можливими шляхами робочого каталогу.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    search_dirs = [
        current_dir,
        os.getcwd(),
        os.path.join(os.getcwd(), "lab3"),
        os.path.join(current_dir, "lab3"),
        "/mount/src/test_streamlit/lab3",
        "/mount/src/test_streamlit"
    ]

    model_path = None
    preprocessor_path = None

    for directory in search_dirs:
        m_candidate = os.path.join(directory, "ny_house_price_model.keras")
        p_candidate = os.path.join(directory, "ny_house_preprocessor.pkl")
        if os.path.exists(m_candidate) and os.path.exists(p_candidate):
            model_path = m_candidate
            preprocessor_path = p_candidate
            break

    if model_path is None or preprocessor_path is None:
        st.error(
            "❌ Не знайдено файли артефактів моделі (`ny_house_price_model.keras` або `ny_house_preprocessor.pkl`)!\n\n"
            "Переконайтеся, що обидва файли завантажені у репозиторій GitHub."
        )
        st.stop()

    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_artifacts()

# 4. Бічна панель: інформація про застосунок
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&auto=format&fit=crop&q=60",
        caption="New York City Real Estate",
        use_container_width=True
    )
    st.title("Про проєкт")
    st.markdown(
        """
        Цей застосунок використовує **глибоку нейронну мережу (MLP)**, 
        навчену на даних ринку житла Нью-Йорка, для оцінки вартості об'єктів.
        
        **Особливості:**
        * Географічні координати автоматично прив'язуються до обраного району.
        * Можливість ручного коригування GPS за потреби.
        * Зворотне перетворення логарифмічного масштабу ціни в реальні долари.
        """
    )
    st.divider()
    st.caption("Лабораторна робота № 3 | Кафедра ІПЗ")

# 5. Основний інтерфейс: Форма параметрів об'єкта
st.title("🏠 Оцінка вартості житла у Нью-Йорку за допомогою ШІ")
st.markdown(
    "Оберіть характеристики нерухомості та район розташування. "
    "Географічні координати для моделі та карти буде визначено автоматично."
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📐 Характеристики об'єкта")
    
    property_type = st.selectbox(
        "Тип об'єкта (Type)",
        options=[
            "Condo for sale",
            "House for sale",
            "Co-op for sale",
            "Townhouse for sale",
            "Multi-family home for sale"
        ]
    )
    
    sqft = st.number_input(
        "Площа житла (кв. фути)",
        min_value=200.0,
        max_value=20000.0,
        value=1500.0,
        step=50.0
    )

    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        beds = st.slider("Спальні (Beds)", min_value=1, max_value=10, value=3, step=1)
    with col_sub2:
        bath = st.number_input("Санвузли (Bath)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

with col_right:
    st.subheader("📍 Розташування та район")
    
    # Вибір субрайону
    subloc = st.selectbox(
        "Район (Sublocality)",
        options=list(DISTRICT_DATA.keys()),
        index=0
    )

    # Отримуємо типові координати та округ для обраного району
    default_info = DISTRICT_DATA[subloc]
    default_locality = default_info["locality"]
    default_lat = default_info["lat"]
    default_lon = default_info["lon"]

    locality_options = [
        "New York County",
        "Kings County",
        "Queens County",
        "Richmond County",
        "Bronx County",
        "Other"
    ]
    locality_index = locality_options.index(default_locality) if default_locality in locality_options else 5

    locality = st.selectbox(
        "Округ (Locality)",
        options=locality_options,
        index=locality_index
    )

    st.caption(f"ℹ️ *{default_info['desc']}*")

    # Ручне коригування координат приховане у згортку
    with st.expander("🛠️ Точні GPS-координати (налаштовуються автоматично)"):
        st.markdown("За замовчуванням координати встановлені на центр обраного району. Ви можете змінити їх за бажанням:")
        latitude = st.number_input(
            "Широта (Latitude)",
            min_value=40.45,
            max_value=40.95,
            value=default_lat,
            format="%.4f"
        )
        longitude = st.number_input(
            "Довгота (Longitude)",
            min_value=-74.30,
            max_value=-73.65,
            value=default_lon,
            format="%.4f"
        )

# Попередній перегляд розташування на карті
preview_map = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
st.markdown(f"##### Положення обраного району ({subloc}) на карті:")
st.map(preview_map, zoom=11)

# 6. Інференс моделі та розрахунок вартості
st.divider()

if st.button("🚀 Розрахувати орієнтовну вартість", type="primary", use_container_width=True):
    total_rooms = beds + bath
    log_propertysqft = np.log1p(sqft)

    # Формуємо DataFrame з усіма 9 ознаками, на яких навчалася модель
    input_df = pd.DataFrame({
        "BEDS": [beds],
        "BATH": [bath],
        "TOTAL_ROOMS": [total_rooms],
        "LOG_PROPERTYSQFT": [log_propertysqft],
        "LATITUDE": [latitude],
        "LONGITUDE": [longitude],
        "TYPE": [property_type],
        "SUBLOC_CLEAN": [subloc],
        "LOCALITY_CLEAN": [locality]
    })

    with st.spinner("Нейромережа обчислює прогноз ринкової вартості..."):
        try:
            processed_input = preprocessor.transform(input_df)
            pred_log_price = model.predict(processed_input, verbose=0)
            predicted_price = float(np.expm1(pred_log_price[0][0]))

            st.success("✅ Розрахунок успішно виконано!")

            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric(
                    label="Орієнтовна вартість",
                    value=f"${predicted_price:,.2f}"
                )
            with res2:
                price_per_sqft = predicted_price / sqft
                st.metric(
                    label="Ціна за кв. фут",
                    value=f"${price_per_sqft:,.2f} / sqft"
                )
            with res3:
                st.metric(
                    label="Локація об'єкта",
                    value=f"{subloc}",
                    delta=locality
                )

        except Exception as e:
            st.error(f"Помилка під час обчислення: {e}")