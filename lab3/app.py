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

# 2. Кешоване завантаження моделі та препроцесора
@st.cache_resource
def load_artifacts():
    """
    Завантажує натреновану Keras-модель та scikit-learn препроцесор.
    Перевіряє як папку скрипта (lab3), так і поточну робочу директорію.
    """
    # 1. Перевірка шляху всередині папки скрипта
    model_path = os.path.join(BASE_DIR, "ny_house_price_model.keras")
    preprocessor_path = os.path.join(BASE_DIR, "ny_house_preprocessor.pkl")

    # 2. Якщо не знайдено, перевіряємо поточну директорію запуску
    if not os.path.exists(model_path):
        model_path = "ny_house_price_model.keras"
    if not os.path.exists(preprocessor_path):
        preprocessor_path = "ny_house_preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error(
            f"❌ Не знайдено файли артефактів моделі!\n\n"
            f"- Шукали за шляхом: `{os.path.join(BASE_DIR, 'ny_house_price_model.keras')}`\n"
            f"- Поточна робоча папка: `{os.getcwd()}`\n\n"
            f"**Як виправити:**\n"
            f"1. Переконайтеся, що файли `ny_house_price_model.keras` та `ny_house_preprocessor.pkl` "
            f"завантажені на GitHub у папку `lab3/` (перевірте, чи вони не заблоковані у `.gitignore`).\n"
            f"2. Зробіть `git add`, `git commit` та `git push` цих двох файлів."
        )
        st.stop()

    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_artifacts()

# 3. Бічна панель: інформація про застосунок та автора
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&auto=format&fit=crop&q=60",
        caption="New York City Housing",
        use_container_width=True
    )
    st.title("Про проєкт")
    st.markdown(
        """
        Цей веб-застосунок використовує **глибоку нейронну мережу (MLP)**, 
        навчену на даних ринку нерухомості Нью-Йорка, для оцінки ринкової 
        вартості житлових об'єктів.
        
        **Стек технологій:**
        * **Фреймворк:** Streamlit
        * **Модель:** TensorFlow / Keras (MLP Regressor)
        * **Препроцесинг:** Scikit-Learn `ColumnTransformer`
        """
    )
    st.divider()
    st.caption("Лабораторна робота № 3 | Кафедра ІПЗ")

# 4. Основна панель: Заголовок та вхідні форми
st.title(" Оцінка вартості житла у Нью-Йорку за допомогою ШІ")
st.markdown(
    "Введіть фізичні та географічні характеристики нерухомості для отримання миттєвого розрахунку орієнтовної ціни."
)
st.subheader(" Параметри нерухомості")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("##### 📐 Фізичні розміри")
    beds = st.slider("Кількість спалень (Beds)", min_value=1, max_value=10, value=3, step=1)
    bath = st.number_input("Кількість санвузлів (Bath)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    sqft = st.number_input("Площа (кв. фути)", min_value=200.0, max_value=20000.0, value=1500.0, step=50.0)
with col2:
    st.markdown("##### 🏢 Категорія та район")
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
    subloc = st.selectbox(
        "Субрайон (Sublocality)",
        options=[
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Staten Island",
            "The Bronx",
            "Flushing",
            "Riverdale",
            "Coney Island",
            "Other"
        ]
    )

    locality = st.selectbox(
        "Округ (Locality)",
        options=[
            "New York County",
            "Kings County",
            "Queens County",
            "Richmond County",
            "Bronx County",
            "Other"
        ]
    )
with col3:
    st.markdown("##### 📍 Геолокація")
    latitude = st.number_input("Широта (Latitude)", min_value=40.45, max_value=40.95, value=40.7128, format="%.4f")
    longitude = st.number_input("Довгота (Longitude)", min_value=-74.30, max_value=-73.65, value=-74.0060, format="%.4f")
    st.info("💡 Мангеттен приблизно: 40.75° N, -73.98° W")

# 5. Інженерія ознак, передобробка та інференс моделі 
st.divider()

if st.button(" Розрахувати орієнтовну вартість", type="primary", use_container_width=True):
    # Розрахунок кумулятивних ознак відповідно до конвеєра Лабораторної №1
    total_rooms = beds + bath
    log_propertysqft = np.log1p(sqft)

    # Формування єдиного датафрейму для трансформера
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

    with st.spinner("Нейромережа обчислює прогноз..."):
        try:
            # Препроцесинг за допомогою збереженого ColumnTransformer
            processed_input = preprocessor.transform(input_df)

            # Прямий прохід моделі Keras (прогноз у масштабі log-ціни)
            pred_log_price = model.predict(processed_input, verbose=0)

            # Зворотне перетворення log1p -> реальна вартість
            predicted_price = float(np.expm1(pred_log_price[0][0]))

            # Відображення результатів користувачеві
            st.success(" Розрахунок успішно завершено!")
            
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.metric(
                    label="Прогнозована ринкова вартість",
                    value=f"${predicted_price:,.2f}"
                )
            with res_col2:
                price_per_sqft = predicted_price / sqft
                st.metric(
                    label="Орієнтовна ціна за кв. фут",
                    value=f"${price_per_sqft:,.2f} / sqft"
                )

            # Карта розміщення об'єкта
            map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
            st.markdown("##### Розташування об'єкта на карті Нью-Йорка:")
            st.map(map_data, zoom=11)

        except Exception as e:
            st.error(f"Помилка під час обробки даних або виконання прогнозу: {e}")