import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.title("🏖️ Weekend Getaway Ranker")

df = pd.read_csv("Top Indian Places to Visit.csv")

df = df[
    ["City", "Name",
     "Google review rating",
     "Number of google review in lakhs",
     "time needed to visit in hrs"]
].dropna()

distance_map = {
    "Kolkata": {
        "Digha": 183,
        "Shantiniketan": 165,
        "Darjeeling": 611
    },
    "Delhi": {
        "Jaipur": 281,
        "Agra": 233,
        "Rishikesh": 242
    },
    "Bengaluru": {
        "Mysuru": 143,
        "Coorg": 252,
        "Chikmagalur": 245
    }
}

def rank_places(source_city):
    if source_city not in distance_map:
        return None

    df_city = df[df["City"] != source_city].copy()
    df_city["distance_km"] = df_city["City"].map(distance_map[source_city])
    df_city = df_city.dropna(subset=["distance_km"])

    scaler = MinMaxScaler()
    df_city["rating_norm"] = scaler.fit_transform(
        df_city[["Google review rating"]]
    )
    df_city["popularity_norm"] = scaler.fit_transform(
        df_city[["Number of google review in lakhs"]]
    )

    df_city["final_score"] = (
        0.5 * df_city["rating_norm"]
        + 0.3 * df_city["popularity_norm"]
        + 0.2 * (1 / (df_city["distance_km"] + 1))
    )

    return df_city.sort_values("final_score", ascending=False).head(5)


source_city = st.selectbox(
    "Select Source City",
    list(distance_map.keys())
)

if st.button("Recommend"):
    result = rank_places(source_city)
    if result is not None:
        st.dataframe(
            result[["Name", "City", "distance_km", "final_score"]]
        )
    else:
        st.error("Source city not supported yet.")
