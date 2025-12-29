import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("="*80)
print("WEEKEND GETAWAY RANKER - TOP INDIAN DESTINATIONS")
print("="*80)

# Load dataset
df = pd.read_csv("Top Indian Places to Visit.csv")

# Required columns
df = df[
    ["City", "Name",
     "Google review rating",
     "Number of google review in lakhs",
     "time needed to visit in hrs"]
].dropna()

# Distance map (extendable)
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

def rank_places(source_city, top_n=5):
    if source_city not in distance_map:
        raise ValueError(
            f"Source city '{source_city}' not supported yet."
        )

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

    df_city["time_score"] = df_city["time needed to visit in hrs"].apply(
        lambda x: 1 if x <= 48 else 0.5
    )
    df_city["distance_score"] = 1 / (df_city["distance_km"] + 1)

    df_city["final_score"] = (
        0.4 * df_city["rating_norm"]
        + 0.3 * df_city["popularity_norm"]
        + 0.2 * df_city["distance_score"]
        + 0.1 * df_city["time_score"]
    )

    return df_city.sort_values(
        "final_score", ascending=False
    ).head(top_n)


# -------- USER INPUT --------
source_city = input("\nEnter source city: ").strip()

try:
    result = rank_places(source_city)
    print(f"\nTop weekend destinations from {source_city}:\n")
    print(result[["Name", "City", "distance_km", "final_score"]])
except ValueError as e:
    print("❌", e)
