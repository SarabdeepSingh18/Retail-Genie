import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
# from surprise import Dataset, Reader


# Page Config
st.set_page_config(page_title="Retail Genie", layout="wide")

# Sidebar Navigation
st.sidebar.title("🧞‍♂️ Retail Genie")
section = st.sidebar.radio("✨ Choose a module:", [
    "🏠 Overview",
    "📈 Sales Forecasting",
    "🔮 Customer Churn Prediction",
    "👥 Customer Segmentation",
    "🛍️ Product Recommendation"
])

# ---- 1. Overview ----
if section == "🏠 Overview":
    st.title("🧞‍♂️ Retail Genie - Smart Retail AI Assistant")
    st.markdown("""
    Welcome to **Retail Genie**! 🛒 This tool brings AI to your fingertips to:
    - 📈 Forecast future sales
    - 🔮 Predict customer churn
    - 👥 Segment customers using RFM
    - 🛍️ Recommend personalized products

    Select a module from the sidebar to get started!
    """)

# ---- 2. Sales Forecasting ----
elif section == "📈 Sales Forecasting":
    st.header("📊 Sales Forecasting using Manual ETS")
    st.markdown("""
    Upload your daily sales data or use the sample provided.  
    Your file must contain a **Date column** named `InvoiceDate` and a **Sales column**.  
    This tool uses **Triple Exponential Smoothing (ETS)** to forecast future sales.
    """)

    # 1. Upload or load sample
    file = st.file_uploader("📂 Upload daily sales CSV or Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file, parse_dates=["InvoiceDate"], index_col="InvoiceDate")
        st.success("✅ File uploaded successfully.")
    else:
        st.info("ℹ️ Using sample dataset: `retail1.csv`")
        df = pd.read_csv("retail1.csv", parse_dates=["InvoiceDate"], index_col="InvoiceDate")

    st.write("🧾 Sample of your Sales Data:")
    st.dataframe(df.head())

    # 2. Define parameters
    st.markdown("### ⚙️ Configure Forecasting Parameters")

    alpha = st.slider("α: Level smoothing", 
                      min_value=0.01, max_value=1.0, value=0.2, step=0.01,
                      help="Controls how quickly the model adapts to changes in the overall sales level.")

    beta = st.slider("β: Trend smoothing", 
                     min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                     help="Controls how quickly the model adapts to trends over time.")

    gamma = st.slider("γ: Seasonal smoothing", 
                      min_value=0.01, max_value=1.0, value=0.05, step=0.01,
                      help="Controls how quickly seasonal patterns are updated.")

    n_preds = st.slider("📅 Forecast horizon (in days)", 
                        min_value=1, max_value=60, value=30,
                        help="How many days into the future to predict sales.")

    st.markdown("These values control how much emphasis the model gives to recent data versus older trends and seasonal patterns.")

    # 3. Helper functions
    def initial_trend(series, slen):
        return sum((series[i + slen] - series[i]) / slen for i in range(slen)) / slen

    def initial_seasonal_components(series, slen):
        n_seasons = len(series) // slen
        season_averages = [series[slen*j : slen*j+slen].mean() for j in range(n_seasons)]
        seasonals = {}
        for i in range(slen):
            seasonals[i] = sum(series[slen*j + i] - season_averages[j] for j in range(n_seasons)) / n_seasons
        return seasonals

    def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
        seasonals = initial_seasonal_components(series, slen)
        level = series[0]
        trend = initial_trend(series, slen)
        result = []

        for i in range(len(series) + n_preds):
            if i == 0:
                result.append(series[0])
                continue
            if i < len(series):
                val = series[i]
                last_level = level
                level = alpha * (val / seasonals[i % slen]) + (1 - alpha) * (level + trend)
                trend = beta * (level - last_level) + (1 - beta) * trend
                seasonals[i % slen] = gamma * (val / level) + (1 - gamma) * seasonals[i % slen]
                result.append((level + trend) * seasonals[i % slen])
            else:
                m = i - len(series) + 1
                result.append((level + m * trend) * seasonals[i % slen])
        return result

    # 4. Apply smoothing
    sales = df["Sales"].astype(float).values
    period = 7  # assuming weekly seasonality
    full_result = triple_exponential_smoothing(sales, period, alpha, beta, gamma, n_preds)

    hist_vals = full_result[:len(sales)]
    forecast_vals = full_result[len(sales):]

    # 5. Build data for plotting
    dates = df.index
    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=n_preds, freq='D')
    fitted_series = pd.Series(hist_vals, index=dates, name="Fitted")
    forecast_series = pd.Series(forecast_vals, index=future_dates, name="Forecast")

    # 6. Plot
    st.subheader(f"📈 Historical vs Forecasted Sales (next {n_preds} days)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, sales, label="Actual", linewidth=2)
    ax.plot(fitted_series, label="Fitted", linestyle="--")
    ax.plot(forecast_series, label="Forecast", linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Sales Forecasting completed using ETS model!")




# ---- 3. Customer Churn ----
elif section == "🔮 Customer Churn Prediction":
    st.header("🔍 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability.")

    churn_model = joblib.load("churn_model.pkl")
    scalers = {
        'Tenure': joblib.load("scalersCHURN/Tenure.pkl"),
        'CityTier': joblib.load("scalersCHURN/CityTier.pkl"),
        'WarehouseToHome': joblib.load("scalersCHURN/WarehouseToHome.pkl"),
        'HourSpendOnApp': joblib.load("scalersCHURN/HourSpendOnApp.pkl"),
        'NumberOfDeviceRegistered': joblib.load("scalersCHURN/NumberOfDeviceRegistered.pkl"),
        'SatisfactionScore': joblib.load("scalersCHURN/SatisfactionScore.pkl"),
        'NumberOfAddress': joblib.load("scalersCHURN/NumberOfAddress.pkl"),
        'Complain': joblib.load("scalersCHURN/Complain.pkl"),
        'OrderAmountHikeFromlastYear': joblib.load("scalersCHURN/OrderAmountHikeFromlastYear.pkl"),
        'CouponUsed': joblib.load("scalersCHURN/CouponUsed.pkl"),
        'OrderCount': joblib.load("scalersCHURN/OrderCount.pkl"),
        'DaySinceLastOrder': joblib.load("scalersCHURN/DaySinceLastOrder.pkl"),
        'CashbackAmount': joblib.load("scalersCHURN/CashbackAmount.pkl")
    }

    churn_df = pd.read_excel('E Commerce Dataset.xlsx', sheet_name=1)

    with st.form("churn_form"):
        tenure = st.slider("📅 Tenure (in months)", 0, int(churn_df['Tenure'].max()), 12)
        preferred_device = st.selectbox("📱 Preferred Login Device", ['Mobile Phone', 'Computer'])
        city_tier = st.selectbox("🌆 City Tier", sorted(churn_df['CityTier'].unique()))
        warehouse_to_home = st.slider("🚚 Warehouse To Home (in km)", 0, int(churn_df['WarehouseToHome'].max()), 10)
        payment_mode = st.selectbox("💳 Preferred Payment Mode", ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet', 'COD'])
        gender = st.selectbox("🧑‍🤝‍🧑 Gender", ['Female', 'Male'])
        hour_spent = st.slider("⏱️ Hours Spent on App", 0, int(churn_df['HourSpendOnApp'].max()), 2)
        devices_registered = st.slider("📱 Number of Devices Registered", 1, int(churn_df['NumberOfDeviceRegistered'].max()), 2)
        preferred_order_cat = st.selectbox("📦 Preferred Order Category", ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
        satisfaction = st.slider("😊 Satisfaction Score", 1, int(churn_df['SatisfactionScore'].max()), 3)
        marital_status = st.selectbox("💍 Marital Status", ['Single', 'Divorced', 'Married'])
        addresses = st.slider("🏠 Number of Addresses", 1, int(churn_df['NumberOfAddress'].max()), 2)
        complain = st.selectbox("❗ Complain Registered", [0, 1])
        hike = st.slider("📈 Order Amount Hike From Last Year", 0.0, float(churn_df['OrderAmountHikeFromlastYear'].max()), 10.0)
        coupon_used = st.slider("🎟️ Coupons Used", 0, int(churn_df['CouponUsed'].max()), 1)
        order_count = st.slider("🛒 Order Count", 0, int(churn_df['OrderCount'].max()), 10)
        days_since_last = st.slider("📆 Days Since Last Order", 0, int(churn_df['DaySinceLastOrder'].max()), 30)
        cashback = st.slider("💰 Cashback Amount", 0.0, float(churn_df['CashbackAmount'].max()), 50.0)

        submitted = st.form_submit_button("🧠 Predict")

    if submitted:
        preferred_device_map = {'Mobile Phone': 1, 'Computer': 0}
        payment_mode_map = {'Debit Card': 3, 'UPI': 5, 'Credit Card': 2, 'Cash on Delivery': 1, 'E wallet': 4, 'COD': 0}
        gender_map = {'Female': 0, 'Male': 1}
        order_cat_map = {'Laptop & Accessory': 2, 'Mobile Phone': 3, 'Others': 4, 'Fashion': 0, 'Grocery': 1}
        marital_map = {'Single': 2, 'Divorced': 0, 'Married': 1}

        input_dict = {
            'Tenure': tenure,
            'PreferredLoginDevice': preferred_device_map[preferred_device],
            'CityTier': city_tier,
            'WarehouseToHome': warehouse_to_home,
            'PreferredPaymentMode': payment_mode_map[payment_mode],
            'Gender': gender_map[gender],
            'HourSpendOnApp': hour_spent,
            'NumberOfDeviceRegistered': devices_registered,
            'PreferedOrderCat': order_cat_map[preferred_order_cat],
            'SatisfactionScore': satisfaction,
            'MaritalStatus': marital_map[marital_status],
            'NumberOfAddress': addresses,
            'Complain': complain,
            'OrderAmountHikeFromlastYear': hike,
            'CouponUsed': coupon_used,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_since_last,
            'CashbackAmount': cashback
        }

        input_df = pd.DataFrame([input_dict])
        for col in input_df.columns:
            if col in scalers:
                input_df[col] = scalers[col].transform(input_df[[col]])

        prob = churn_model.predict_proba(input_df)[0][1]
        result = churn_model.predict(input_df)[0]

        st.markdown(f"### 🧾 Churn Prediction: {'✅ Yes' if result == 1 else '❌ No'}")
        st.progress(int(prob * 100))
        st.info(f"📊 Churn Probability: {prob:.2%}")

# ---- 4. Customer Segmentation ----
elif section == "👥 Customer Segmentation":
    st.header("👥 Customer Clustering (Premium vs Non-Premium)")
    st.markdown("Upload transaction data to segment customers.")

    cluster_model = joblib.load("KMean_clust.pkl")
    file = st.file_uploader("📁 Upload transaction data", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        st.write(df.head())

        if st.button("🔍 Segment Customers"):
            try:
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

                Latest_Date = dt.datetime(2011, 12, 10)
                rfm_df = df.groupby('CustomerID').agg({
                    'InvoiceDate': lambda x: (Latest_Date - x.max()).days,
                    'InvoiceNo': lambda x: len(x),
                    'TotalAmount': lambda x: x.sum()
                })

                rfm_df['InvoiceDate'] = rfm_df['InvoiceDate'].astype(int)
                rfm_df.rename(columns={
                    'InvoiceDate': 'Recency',
                    'InvoiceNo': 'Frequency',
                    'TotalAmount': 'Monetary'
                }, inplace=True)
                def handle_neg_n_zero(num):
                    if num <= 0:
                        return 1
                    else:
                        return num
                #Applying handle_neg_n_zero function to Recency and Monetary columns 
                rfm_df['Recency'] = [handle_neg_n_zero(x) for x in rfm_df.Recency]
                rfm_df['Monetary'] = [handle_neg_n_zero(x) for x in rfm_df.Monetary]

    

                rfm_log = np.log1p(rfm_df)
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm_log)

                labels = cluster_model.predict(rfm_scaled)
                rfm_df['Segment'] = labels

                st.success("✅ Segmentation Completed!")
                st.write(rfm_df)

                # 🎨 Segment Distribution
                st.subheader("📊 Segment Distribution")
                fig1, ax1 = plt.subplots()
                rfm_df['Segment'].value_counts().plot(kind='bar', ax=ax1, color=['#FF6F61', '#6B5B95'])
                ax1.set_xlabel("Segment")
                ax1.set_ylabel("Count")
                ax1.set_title("Customer Segment Count")
                st.pyplot(fig1)

                # 📈 Boxplots
                st.subheader("📈 RFM Features by Segment")
                fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
                for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
                    rfm_df.boxplot(column=col, by='Segment', ax=ax2[i])
                    ax2[i].set_title(f'{col} by Segment')
                    ax2[i].set_ylabel(col)
                plt.tight_layout()
                st.pyplot(fig2)

                # 📊 Pie Chart for Premium vs Non-Premium Distribution
                st.subheader("🍰 Non-Premium vs Premium Distribution")
                segment_counts = rfm_df['Segment'].value_counts()
                fig3, ax3 = plt.subplots()
                ax3.pie(segment_counts, labels=["Non-Premium 🧺", "Premium 🏆"], autopct='%1.1f%%', colors=['#FF6F61', '#6B5B95'], startangle=90)
                ax3.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
                ax3.set_title("Customer Segmentation (Premium vs Non-Premium)")
                st.pyplot(fig3)

               
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---- 5. Product Recommendation ----
elif section == "🛍️ Product Recommendation":

    st.header("🛍️ Item-Based Product Recommendation")

    product_id = st.text_input("📦 Enter Product ID you liked")

    @st.cache_resource
    def load_recommendation_data():
        
        # Load the dataset
        new_df = joblib.load("RecomDF.pkl")  # Assumes the original new_df saved
        new_df1 = new_df.head(10000)

        # Create ratings matrix
        ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
        X = ratings_matrix.T  # Transpose to get products x users

        # SVD Decomposition
        SVD = TruncatedSVD(n_components=10)
        decomposed_matrix = SVD.fit_transform(X)

        # Correlation Matrix
        correlation_matrix = np.corrcoef(decomposed_matrix)
        product_names = list(X.index)

        return correlation_matrix, product_names, X

    correlation_matrix, product_names, X = load_recommendation_data()

    def recommend_similar_products(product_id, correlation_matrix, product_names, X, threshold=0.65, top_n=10):
        if product_id not in product_names:
            return None

        product_idx = product_names.index(product_id)
        correlation_product_ID = correlation_matrix[product_idx]

        recommended_ids = list(X.index[correlation_product_ID > threshold])
        if product_id in recommended_ids:
            recommended_ids.remove(product_id)

        return recommended_ids[:top_n]

    if st.button("🎯 Get Recommendations"):
        recommendations = recommend_similar_products(product_id, correlation_matrix, product_names, X)

        if recommendations is not None and len(recommendations) > 0:
            st.success(f"🎁 Top Recommendations based on Product `{product_id}`:")
            for idx, item in enumerate(recommendations, 1):
                st.write(f"{idx}. 📦 Product ID: `{item}`")
        elif recommendations == []:
            st.warning("🤷 No highly correlated products found. Try another Product ID.")
        else:
            st.error("🚫 Product ID not found. Please enter a valid Product ID.")