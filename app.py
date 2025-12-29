import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Weekend Getaway Ranker", layout="wide")
st.title("🏖️ Weekend Getaway Ranker (All Cities Supported)")

# Load data
df = pd.read_csv("Top Indian Places to Visit.csv")

df = df[
    ["Zone", "State", "City", "Name",
     "Google review rating",
     "Number of google review in lakhs",
     "time needed to visit in hrs"]
].dropna()

# Source city selector (ALL cities)
source_city = st.selectbox(
    "Select Source City",
    sorted(df["City"].unique())
)

source_row = df[df["City"] == source_city].iloc[0]
source_state = source_row["State"]
source_zone = source_row["Zone"]

def compute_distance_score(row):
    if row["City"] == source_city:
        return 0
    if row["State"] == source_state:
        return 1.0
    if row["Zone"] == source_zone:
        return 0.7
    return 0.4

df["distance_score"] = df.apply(compute_distance_score, axis=1)

# Normalize scores
scaler = MinMaxScaler()
df["rating_norm"] = scaler.fit_transform(df[["Google review rating"]])
df["popularity_norm"] = scaler.fit_transform(
    df[["Number of google review in lakhs"]]
)

df["time_score"] = df["time needed to visit in hrs"].apply(
    lambda x: 1 if x <= 48 else 0.5
)

df["final_score"] = (
    0.4 * df["rating_norm"]
    + 0.3 * df["popularity_norm"]
    + 0.2 * df["distance_score"]
    + 0.1 * df["time_score"]
)

result = (
    df[df["City"] != source_city]
    .sort_values("final_score", ascending=False)
    .head(5)
)

st.subheader(f"Top Weekend Destinations from {source_city}")
st.dataframe(
    result[["Name", "City", "State", "final_score"]],
    use_container_width=True
)
