



























# Import necessary libraries
import streamlit as st
import pandas as pd

# Add the file uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Check if the file is uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Display the DataFrame in the Streamlit app
    st.write(df)































import streamlit as st
import pandas as pd

# Data dictionary (example)
data_dict = {
    "Column Name": ["From Date", "To Date", "Ozone", "CO", "SO2", "NO2", "PM10", "PM2.5", "State", "City", "Station"],
    "Description": [
        "The starting date and time of the observation",
        "The ending date and time of the observation",
        "The ozone concentration level in the air",
        "The carbon monoxide concentration level in the air",
        "The sulfur dioxide concentration level in the air",
        "The nitrogen dioxide concentration level in the air",
        "The concentration of particulate matter smaller than 10 micrometers",
        "The concentration of particulate matter smaller than 2.5 micrometers",
        "The state where the monitoring station is located",
        "The city where the monitoring station is located",
        "The ID of the air quality monitoring station"
    ],
    "Data Type": ["Datetime", "Datetime", "Float", "Float", "Float", "Float", "Float", "Float", "String", "String", "String"],
    "Units": ["-", "-", "µg/m³", "µg/m³", "µg/m³", "µg/m³", "µg/m³", "µg/m³", "-", "-", "-"]
}

# Create a DataFrame for the data dictionary
df_dict = pd.DataFrame(data_dict)

# Display the Data Dictionary in Streamlit
st.write("### Data Dictionary for Air Quality Data")
st.write(df_dict)





from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns






import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import streamlit as st

import streamlit as st
import pandas as pd

# Load the data
df = pd.read_excel("AndhraPradesh.xlsx")



# --- KNN Classification for Pollution Levels ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Example: using PM2.5 as the feature to classify pollution as 'High' or 'Low'
pollutant = 'PM2.5'  # you can make this selectable later with st.selectbox

if pollutant in df.columns:
    st.subheader(f"KNN Classification: {pollutant} Levels")

    # Drop rows with missing values for the selected pollutant
    df_knn = df[[pollutant]].dropna().copy()

    # Create a binary target: High if value > mean, Low otherwise
    df_knn['Target'] = df_knn[pollutant].apply(lambda x: 1 if x > df_knn[pollutant].mean() else 0)

    # Features and target
    X = df_knn[[pollutant]]
    y = df_knn['Target']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Predict and Evaluate
    y_pred = knn.predict(X_test_scaled)
    st.write("### Accuracy:", accuracy_score(y_test, y_pred))
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix for {pollutant} KNN Classifier")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.warning(f"Pollutant '{pollutant}' not found in dataset.")








import pandas as pd
import numpy as np

# Load the Excel dataset
data = pd.read_excel("AndhraPradesh.xlsx")  # Make sure the file is in the same folder


# Show the columns so you know what exists
st.write("Available columns in the dataset:")
st.write(list(data.columns))

# Let the user pick which column to predict
target_column = st.selectbox(
    "Select the target column (the value you want to predict):",
    options=list(data.columns)
)

# Features = all columns except the target
feature_cols = [c for c in data.columns if c != target_column]

# Use only numeric feature columns (to avoid issues with text)
X_df = data[feature_cols].select_dtypes(include=["number"]).copy()

# Handle case where there are no numeric features
if X_df.shape[1] == 0:
    st.error("No numeric feature columns found after excluding the target column.")
    st.stop()

# Fill missing numeric values (simple strategy)
X_df = X_df.fillna(0)

# Final feature and target arrays
X = X_df.values
y = data[target_column].values






# Gaussian Naive Bayes implementation
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9  # Avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_probability(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._gaussian_probability(c, x)))
            posteriors[c] = prior + likelihood
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

# Train and predict
nb = GaussianNaiveBayes()
nb.fit(X, y)
predictions = nb.predict(X)

# Add predictions to your DataFrame
data["predictions"] = predictions

# Optional: show accuracy
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy:.2f}")






















# Ensure 'From Date' and 'To Date' are in datetime format with the correct format
df['From Date'] = pd.to_datetime(df['From Date'], format="%d-%m-%Y %H:%M")
df['To Date'] = pd.to_datetime(df['To Date'], format="%d-%m-%Y %H:%M")

# Date range selection from sidebar
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", 
    [df['From Date'].min(), df['From Date'].max()]
)


# --- City Selector Sidebar ---
cities = df['City'].unique()  # Get unique cities from dataset
selected_city = st.sidebar.selectbox("Select City", cities)

# Filter the DataFrame by selected city
df_city = df[df['City'] == selected_city]
st.write(f"### Data for {selected_city}")



# Filter by date after city selection
df_filtered = df_city[(df_city['From Date'] >= pd.to_datetime(start_date)) &
                      (df_city['To Date'] <= pd.to_datetime(end_date))]


# Display the filtered data range
st.write(f"Displaying data from {start_date} to {end_date}")


# --- Pollutant Selector Sidebar ---
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO2', 'Ozone', 'SO2']
available_pollutants = [p for p in pollutants if p in df_filtered.columns]

selected_pollutant = st.sidebar.selectbox("Select Pollutant", available_pollutants)

# Display average value
avg_value = df_filtered[selected_pollutant].mean()
st.metric(label=f"Average {selected_pollutant}", value=f"{avg_value:.2f}")

# Line chart for selected pollutant
fig_pollutant = px.line(df_filtered, x='From Date', y=selected_pollutant,
                        title=f"{selected_pollutant} Levels Over Time",
                        labels={selected_pollutant: f"{selected_pollutant} Level",
                                'From Date': 'Date'})
st.plotly_chart(fig_pollutant)


# Explanation for the pollutant over time graph
st.markdown("### How to read this graph")
st.markdown(f"- The x-axis shows the date.")
st.markdown(f"- The y-axis shows the {selected_pollutant} concentration level (µg/m³ for PM, ppm for gases).")
st.markdown(f"- Each point/bar represents the pollutant level on that day.")
st.markdown(f"- Higher values indicate worse air quality for that pollutant.")
st.markdown(f"- Average level for this range: {avg_value:.2f}")






from sklearn.cluster import KMeans
import plotly.express as px

# Select columns for clustering
cluster_columns = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2'] if col in df_filtered.columns]

# Keep rows without NaNs for clustering
df_cluster = df_filtered[cluster_columns].dropna()

# Perform K-Means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(df_cluster)

# Add a new 'Cluster' column only for the rows used in clustering
df_filtered.loc[df_cluster.index, 'Cluster'] = clusters

# Display cluster distribution
st.write("### Pollution Hotspot Clusters")
st.dataframe(df_filtered[['From Date', 'City', 'Cluster'] + cluster_columns].head(20))

# Visualize clusters for first two features (PM2.5 vs PM10)
fig_cluster = px.scatter(
    df_filtered.dropna(subset=['Cluster']),
    x='PM2.5',
    y='PM10',
    color='Cluster',
    hover_data=cluster_columns,
    title="K-Means Pollution Hotspot Clusters"
)
st.plotly_chart(fig_cluster)


# Cluster explanation for users
st.markdown("### How to read the pollution hotspot clusters")
st.markdown("- Each point represents a day or station depending on your data filter.")
st.markdown("- Color indicates the pollution cluster:")
st.markdown("  -  (Cluster 2): High pollution")
st.markdown("  -  (Cluster 1): Medium pollution")
st.markdown("  -  (Cluster 0): Low pollution")
st.markdown("- Hover over a point to see exact pollutant levels for that day/station.")
st.markdown("- Clusters are relative within the selected city and date range.")














# --- AQI Calculation Function ---
def calculate_aqi(pm25, pm10, no2, so2, co2=None, ozone=None):
    """
    Simple AQI approximation using PM2.5, PM10, NO2, SO2.
    Adjust the weights as needed.
    """
    aqi = 0.5 * pm25 + 0.2 * pm10 + 0.2 * no2 + 0.1 * so2
    return aqi

# --- Compute AQI for filtered data ---
df_filtered['AQI'] = df_filtered.apply(lambda row: calculate_aqi(
    row.get('PM2.5', 0),
    row.get('PM10', 0),
    row.get('NO2', 0),
    row.get('SO2', 0)
), axis=1)

# --- Average AQI for selected range ---
avg_aqi = df_filtered['AQI'].mean()

# --- Determine color based on AQI ---
def aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 150:
        return "orange"
    elif aqi <= 200:
        return "red"
    elif aqi <= 300:
        return "purple"
    else:
        return "maroon"

# --- Display AQI metric ---
st.markdown(f"### Average AQI: <span style='color:{aqi_color(avg_aqi)}'>{avg_aqi:.2f}</span>", unsafe_allow_html=True)


# --- AQI Reference Table ---
aqi_table = {
    "AQI Range": ["0–50", "51–100", "101–150", "151–200", "201–300", "301+"],
    "Color": ["Green", "Yellow", "Orange", "Red", "Purple", "Maroon"],
    "Pollution Level": ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
}

st.write("### AQI Reference Table")
st.table(aqi_table)








# Check columns in the dataset
st.write("Columns in the dataset:")
st.write(df.columns)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Calculate key statistics
avg_pm25 = df_filtered['PM2.5'].mean()
avg_no2 = df_filtered['NO2'].mean()

# If CO2 column exists, calculate; otherwise skip to avoid errors
if 'CO2' in df_filtered.columns:
    avg_co2 = df_filtered['CO2'].mean()
    max_co2 = df_filtered['CO2'].max()
else:
    avg_co2 = None
    max_co2 = None

max_pm25 = df_filtered['PM2.5'].max()
max_no2 = df_filtered['NO2'].max()

# Display statistics
st.write(f"Average PM2.5: {avg_pm25:.2f} µg/m³")
st.write(f"Average NO2: {avg_no2:.2f} ppm")
if avg_co2:
    st.write(f"Average CO2: {avg_co2:.2f} ppm")

st.write(f"Maximum PM2.5: {max_pm25:.2f} µg/m³")
st.write(f"Maximum NO2: {max_no2:.2f} ppm")
if max_co2:
    st.write(f"Maximum CO2: {max_co2:.2f} ppm")



import plotly.express as px

# --- Line chart for PM2.5 over time ---
fig_pm25 = px.line(df_filtered, x='From Date', y='PM2.5', 
                   title="PM2.5 Levels Over Time",
                   labels={'PM2.5': 'PM2.5 (µg/m³)', 'From Date': 'Date'})
st.plotly_chart(fig_pm25)

# --- Categorize pollution status based on PM2.5 ---
def categorize_pollution(pm25):
    if pm25 <= 50:
        return "Good"
    elif 51 <= pm25 <= 100:
        return "Moderate"
    else:
        return "Hazardous"

df_filtered['Pollution Status'] = df_filtered['PM2.5'].apply(categorize_pollution)

# --- Display pollution status summary ---
st.write("### Pollution Status Summary")
pollution_status_counts = df_filtered['Pollution Status'].value_counts()
st.bar_chart(pollution_status_counts)
st.write(pollution_status_counts)



# --- Line chart for NO2 over time ---
if 'NO2' in df_filtered.columns:
    fig_no2 = px.line(df_filtered, x='From Date', y='NO2',
                      title="NO2 Levels Over Time",
                      labels={'NO2': 'NO2 (ppm)', 'From Date': 'Date'})
    st.plotly_chart(fig_no2)

# --- Line chart for CO2 over time ---
if 'CO2' in df_filtered.columns:
    fig_co2 = px.line(df_filtered, x='From Date', y='CO2',
                      title="CO2 Levels Over Time",
                      labels={'CO2': 'CO2 (ppm)', 'From Date': 'Date'})
    st.plotly_chart(fig_co2)


import seaborn as sns
import matplotlib.pyplot as plt

# --- Select numeric columns for correlation ---
numeric_cols = ['PM2.5', 'PM10', 'NO2']
if 'CO2' in df_filtered.columns:
    numeric_cols.append('CO2')
if 'Ozone' in df_filtered.columns:
    numeric_cols.append('Ozone')
if 'SO2' in df_filtered.columns:
    numeric_cols.append('SO2')

# Compute correlation
corr_matrix = df_filtered[numeric_cols].corr()

# --- Plot heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Pollutants')

# Display the heatmap in Streamlit
st.pyplot(plt)


# Correlation explanation for users
st.markdown("### Correlation between pollutants")
st.markdown("- The heatmap shows how strongly different pollutants are related to each other.")
st.markdown("- Values range from **-1 to 1**:")
st.markdown("  - **1**: Perfect positive correlation (both pollutants increase together)")
st.markdown("  - **0**: No correlation (changes in one pollutant do not affect the other)")
st.markdown("  - **-1**: Perfect negative correlation (one pollutant increases, the other decreases)")
st.markdown("- Example: If PM2.5 and PM10 have a correlation of 0.8, it means when PM2.5 increases, PM10 usually increases too.")
st.markdown("- Use this to understand which pollutants are linked and may share sources (e.g., traffic, industry).")










from sklearn.cluster import KMeans
import numpy as np

# --- Select columns for clustering (numeric pollutants) ---
cluster_cols = ['PM2.5', 'PM10', 'NO2']
if 'CO2' in df_filtered.columns:
    cluster_cols.append('CO2')
if 'Ozone' in df_filtered.columns:
    cluster_cols.append('Ozone')
if 'SO2' in df_filtered.columns:
    cluster_cols.append('SO2')

# Ensure there are no missing values for clustering
df_cluster = df_filtered[cluster_cols].dropna()

# --- Run K-Means ---
k = 3  # Number of clusters: 1-Good, 2-Moderate, 3-Hazardous
kmeans = KMeans(n_clusters=k, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)

# --- Map cluster numbers to labels ---
cluster_labels = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3"}
df_cluster['Cluster Label'] = df_cluster['Cluster'].map(cluster_labels)

# --- Display cluster counts ---
st.write("### Pollution Hotspot Clusters")
st.write(df_cluster['Cluster Label'].value_counts())

# --- Optional: Plot clusters using first two pollutants (2D) ---
fig_cluster = px.scatter(df_cluster, x=cluster_cols[0], y=cluster_cols[1],
                         color='Cluster Label',
                         title="Pollution Hotspot Clusters",
                         labels={cluster_cols[0]: cluster_cols[0],
                                 cluster_cols[1]: cluster_cols[1]})
st.plotly_chart(fig_cluster)


from prophet import Prophet


# Display 30-day forecast
st.subheader("30-Day Forecast of Selected Pollutant")

# --- Forecast PM2.5 Levels ---
if 'PM2.5' in df_filtered.columns:
    st.write("### PM2.5 Forecast")

    # Prepare data for Prophet
    df_prophet = df_filtered[['From Date', 'PM2.5']].rename(columns={'From Date':'ds', 'PM2.5':'y'})
    df_prophet = df_prophet.dropna()

    # Initialize and fit Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    # Make future dataframe for 30 days ahead
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot forecast
    fig_forecast = px.line(forecast, x='ds', y='yhat', title="PM2.5 Forecast for Next 30 Days")
    st.plotly_chart(fig_forecast)

    # Show forecast table
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))






import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import matplotlib.pyplot as plt







st.header("Naive Bayes Classifier")

# 1. Let user choose the target column
target_column = st.selectbox(
    "Select the target column (what you want to predict):",
    options=data.columns,
)

if target_column:
    st.write(f"**Selected target column:** `{target_column}`")

    # 2. Build feature matrix X (numeric columns only, excluding target)
    feature_df = data.drop(columns=[target_column])

    # Keep only numeric columns as features
    X = feature_df.select_dtypes(include=[np.number]).values

    # 3. Prepare target y
    y_raw = data[target_column]

    # If target is numeric, auto-bin into 3 classes: Low / Medium / High
    if pd.api.types.is_numeric_dtype(y_raw):
        st.info(
            "Target column is numeric. "
            "For Naive Bayes (classification), it will be automatically "
            "binned into 3 categories: **Low**, **Medium**, **High**."
        )
        try:
            y_binned = pd.qcut(y_raw, q=3, labels=["Low", "Medium", "High"])
        except ValueError:
            # If there are too few unique values to bin
            y_binned = y_raw.astype(str)
            st.warning(
                "Could not bin the numeric target into 3 groups "
                "(too few unique values). Using raw values as classes instead."
            )
        y = y_binned.astype(str).values
    else:
        # If already categorical / object-like
        y = y_raw.astype(str).values
        st.info("Target column is treated as **categorical classes**.")

    # 4. Handle missing values in X (if any)
    #    (Naive Bayes cannot handle NaNs)
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # 5. Train-Test split
    test_size = st.slider(
        "Test set size (as a fraction):",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,  # keeps class balance
    )

    # 6. Scaling (optional but fine for GaussianNB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Button to train the model
    if st.button("Train Naive Bayes Model"):
        # 7. Train Gaussian Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train_scaled, y_train)

        # 8. Predictions
        y_pred = gnb.predict(X_test_scaled)

        # 9. Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.subheader("Model Performance (Naive Bayes)")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision (weighted):** {prec:.4f}")
        st.write(f"**Recall (weighted):** {rec:.4f}")
        st.write(f"**F1 Score (weighted):** {f1:.4f}")

        # Detailed classification report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # 10. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")

        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, interpolation="nearest")
        ax_cm.figure.colorbar(im, ax=ax_cm)
        ax_cm.set_xticks(np.arange(len(np.unique(y))))
        ax_cm.set_yticks(np.arange(len(np.unique(y))))
        ax_cm.set_xticklabels(np.unique(y), rotation=45, ha="right")
        ax_cm.set_yticklabels(np.unique(y))
        ax_cm.set_ylabel("True label")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_title("Confusion Matrix - Gaussian Naive Bayes")

        # write numbers inside squares
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )

        st.pyplot(fig_cm)

        # 11. ROC Curve & AUC (only if binary classification)
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            st.subheader("ROC Curve")

            # Need probabilities for positive class
            y_proba = gnb.predict_proba(X_test_scaled)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=unique_classes[1])
            auc = roc_auc_score(y_test, y_proba)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve - Gaussian Naive Bayes")
            ax_roc.legend(loc="lower right")

            st.pyplot(fig_roc)
        else:
            st.info(
                "ROC Curve is only shown for **binary classification** problems. "
                f"Current number of classes: {len(unique_classes)}."
            )











# Show filtered data in the app (Optional: You can remove this if you don't want to display it in the app)
st.write(df_filtered)

# Example of visualizing the data (you can replace this with any chart/graph you want)
import plotly.express as px
fig = px.scatter(df_filtered, x='From Date', y='PM2.5', title="PM2.5 over Time")
st.plotly_chart(fig)

# Convert Date column
df['From Date'] = pd.to_datetime(df['From Date'], format='%d-%m-%Y %H:%M')

# Feature Engineering
df['Year'] = df['From Date'].dt.year
df['Month'] = df['From Date'].dt.month
df['Day'] = df['From Date'].dt.day
df['Hour'] = df['From Date'].dt.hour

# Numeric columns
numeric_cols = ['Ozone', 'CO', 'SO2', 'NO2', 'PM10', 'PM2.5']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Scaling
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# AQI calculation (PM2.5 simplified)
def calculate_aqi_pm25(conc):
    if conc <= 12:
        return (50/12)*conc
    elif conc <= 35.4:
        return 50 + (100-51)/(35.4-12)*(conc-12)
    elif conc <= 55.4:
        return 101 + (150-101)/(55.4-35.5)*(conc-35.5)
    elif conc <= 150.4:
        return 151 + (200-151)/(150.4-55.5)*(conc-55.5)
    elif conc <= 250.4:
        return 201 + (300-201)/(250.4-150.5)*(conc-150.5)
    else:
        return 301

df['AQI_PM25'] = df['PM2.5'].apply(lambda x: calculate_aqi_pm25(x))

# AQI Category
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

df['AQI_Category'] = df['AQI_PM25'].apply(categorize_aqi)

# ---------------------------
# Streamlit city selection
st.sidebar.header("Select City")
city_filter = df['City'].unique()
selected_city = st.sidebar.selectbox("Choose a City", city_filter)

df_filtered = df[df['City'] == selected_city]

# PM2.5 Trend
fig1 = px.line(df_filtered, x='From Date', y='PM2.5', title=f"PM2.5 Trend Over Time in {selected_city}")
st.plotly_chart(fig1)

# AQI Category Distribution
aqi_counts = df_filtered['AQI_Category'].value_counts().reset_index()
aqi_counts.columns = ['AQI_Category', 'Count']
fig2 = px.bar(aqi_counts, x='AQI_Category', y='Count', color='AQI_Category', title=f"AQI Category Distribution in {selected_city}")
st.plotly_chart(fig2)

# Regression Prediction Example
features = numeric_cols + ['Year', 'Month', 'Day', 'Hour']
X = df_filtered[features]
y = df_filtered['AQI_PM25']
reg_model = LinearRegression()
reg_model.fit(X, y)
df_filtered.loc[:, 'AQI_Predicted'] = reg_model.predict(X)

fig3 = px.line(df_filtered, x='From Date', y=['AQI_PM25', 'AQI_Predicted'], title=f"Actual vs Predicted AQI_PM25 in {selected_city}")
st.plotly_chart(fig3)

# K-means Clustering for Hotspots
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered.loc[:, 'Pollution_Cluster'] = kmeans.fit_predict(df_filtered[numeric_cols])

fig4 = px.scatter(df_filtered, x='PM10', y='PM2.5', color='Pollution_Cluster',
                  title=f"Pollution Hotspot Clustering in {selected_city}", color_continuous_scale='Viridis')
st.plotly_chart(fig4)



import io

# --- Download Filtered Dataset as CSV ---
csv = df_filtered.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name=f"{selected_city}_air_quality.csv",
    mime="text/csv"
)

# --- Optional: Download PM2.5 Chart as PNG ---
import plotly.io as pio

fig_png = pio.to_image(fig_pollutant, format="png", width=800, height=500)
st.download_button(
    label="Download PM2.5 Chart as PNG",
    data=fig_png,
    file_name=f"{selected_city}_PM25_chart.png",
    mime="image/png"
)

