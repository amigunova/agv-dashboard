# dashboard_live.py
#
# End-to-End Streamlit-App:
# - liest Störungsdaten live aus der Datenbank
# - bereitet Features vor und trainiert Modelle
# - berechnet Risikowerte "Störung in der nächsten Stunde"
# - visualisiert Historie und Prognosen

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------
# Streamlit Grundkonfiguration
# ---------------------------------------------------------------------

st.set_page_config(page_title="AGV-Störungen – Live-Dashboard", layout="wide")

plt.rcParams["figure.figsize"] = (10, 5)

USE_DB = False  # Lokaler Modus: Daten aus Excel, nicht aus der DB
EXCEL_FILE = Path("Stoerungen.xlsx")

# ---------------------------------------------------------------------
# DB-Konfiguration – HIER BEIM KUNDEN ANPASSEN
# ---------------------------------------------------------------------

DB_DIALECT = "mssql+pyodbc"           # oder z.B. "postgresql+psycopg2"
DB_HOST = "SERVERNAME_OR_IP"
DB_PORT = "1433"
DB_NAME = "DATENBANKNAME"
DB_USER = "BENUTZER"
DB_PASSWORD = "PASSWORT"
DB_ODBC_DRIVER = "ODBC Driver 17 for SQL Server"  # bei Bedarf anpassen
DB_TABLE = "SCHEMA.Stoerungen"       # Tabelle oder View mit Störungsdaten

# ---------------------------------------------------------------------
# Modell-Konfiguration
# ---------------------------------------------------------------------

CLUSTER_THRESHOLD_MIN = 5     # Ereignisse < 5 Minuten Abstand werden zu Clustern zusammengefasst
HORIZON_H = 1                 # Vorhersagehorizont: 1 Stunde
MIN_SAMPLES_PER_VEH = 40      # Mindestanzahl Beispiele für fahrzeugspezifisches Modell
MIN_POS_PER_VEH = 5           # Mindestanzahl positiver Beispiele

# ---------------------------------------------------------------------
# Hilfsfunktionen: DB-Verbindung & Daten laden
# ---------------------------------------------------------------------

@st.cache_resource
def get_engine():
    """
    Erzeugt eine SQLAlchemy-Engine für die Datenbankverbindung.
    Wird von Streamlit als Ressource gecached.
    """
    driver_enc = DB_ODBC_DRIVER.replace(" ", "+")
    conn_str = (
        f"{DB_DIALECT}://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?driver={driver_enc}"
    )
    engine = create_engine(conn_str)
    return engine

@st.cache_data(ttl=60)
def fetch_raw_data(lookback_days: int = 30) -> pd.DataFrame:
    """
    Lädt Rohdaten entweder aus Excel (lokal) oder live aus der DB,
    abhängig vom USE_DB-Flag.
    """
    if not USE_DB:
        # Lokaler Modus: einfach Excel laden (komplette Daten)
        df_raw = pd.read_excel(EXCEL_FILE)
        return df_raw

    # --- DB-Modus (für später / auf dem Server) ---
    engine = get_engine()
    start_time = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)

    query = text(f"""
        SELECT
            VehicleNumber,
            EventTime,
            Duration_min,
            EventString,
            StartTime,
            EndTime
        FROM {DB_TABLE}
        WHERE EventTime >= :start_time
    """)

    df_raw = pd.read_sql(query, con=engine, params={"start_time": start_time})
    return df_raw


# ---------------------------------------------------------------------
# Feature Engineering: Vorbereitung + Clustering
# ---------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisiert Spaltennamen (Leerzeichen, Umlaute, ß)."""
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(" ", "_")
          .str.replace("ä", "ae")
          .str.replace("ö", "oe")
          .str.replace("ü", "ue")
          .str.replace("Ä", "Ae")
          .str.replace("Ö", "Oe")
          .str.replace("Ü", "Ue")
          .str.replace("ß", "ss")
    )
    return df


def load_and_prepare_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Grundaufbereitung der Störungsdaten:
    - Spaltennamen normalisieren
    - Zeitspalten parsen
    - Dauer in Minuten berechnen (falls nötig)
    - nur Zeilen mit gültigem EventTime behalten
    """
    df = normalize_columns(df_raw)

    # Zeitspalten
    for col in ["StartTime", "EndTime", "EventTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Dauer in Minuten
    if "Duration_min" in df.columns:
        df["Duration_min"] = pd.to_numeric(df["Duration_min"], errors="coerce")
    elif "Duration_(min)" in df.columns:
        df["Duration_min"] = pd.to_numeric(df["Duration_(min)"], errors="coerce")
    elif "Duration" in df.columns:
        df["Duration_min"] = pd.to_numeric(df["Duration"], errors="coerce")
    elif "StartTime" in df.columns and "EndTime" in df.columns:
        df["Duration_min"] = (df["EndTime"] - df["StartTime"]).dt.total_seconds() / 60
    else:
        df["Duration_min"] = np.nan

    # Nur Zeilen mit EventTime
    df = df[df["EventTime"].notna()].copy()

    return df


def map_event_category(event_str: str) -> str:
    """
    Gruppiert EventString grob in Fehlertyp-Kategorien.
    Diese Kategorien dienen später als 'wahrscheinlichste Ursachen'.
    """
    s = str(event_str)

    if "Handbetrieb" in s:
        return "Handbetrieb"
    if "Beladungsstatus nicht plausibel" in s:
        return "Beladung nicht plausibel"
    if "Klammer Timeout" in s:
        return "Klammer Timeout"
    if "SICK" in s or "Warnfeld" in s:
        return "Sicherheitsfeld / SICK"
    if "Navigation" in s:
        return "Navigation"
    if "Blockung" in s:
        return "Blockung"
    if "Notstopp" in s or "Not-Aus" in s:
        return "Notstopp (sonstige)"
    return "Sonstige"


def add_event_category(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die Spalte EventCategory hinzu."""
    df = df.copy()
    event_col = "EventString" if "EventString" in df.columns else "EvendDescription"
    if event_col not in df.columns:
        # Fallback: leere Kategorie
        df["EventCategory"] = "Sonstige"
    else:
        df["EventCategory"] = df[event_col].apply(map_event_category)
    return df


def cluster_events(df: pd.DataFrame, threshold_min: int = 5) -> pd.DataFrame:
    """
    Fasst aufeinanderfolgende Ereignisse eines Fahrzeugs zusammen,
    wenn der Abstand < threshold_min Minuten ist.
    Ergebnis: Cluster-Level-DataFrame mit EventTime, Dauer, EventCategory.
    """
    df_sorted = df.sort_values(["VehicleNumber", "EventTime"]).copy()

    # Zeitabstand zum vorherigen Ereignis (pro Fahrzeug)
    delta = df_sorted.groupby("VehicleNumber")["EventTime"].diff()
    df_sorted["delta_to_prev"] = delta

    threshold = pd.Timedelta(minutes=threshold_min)
    df_sorted["is_new_cluster"] = (
        df_sorted["delta_to_prev"].isna() | (df_sorted["delta_to_prev"] >= threshold)
    )

    # Cluster-ID pro Fahrzeug
    df_sorted["cluster_id"] = df_sorted.groupby("VehicleNumber")["is_new_cluster"].cumsum()

    def most_frequent(x):
        return x.value_counts().idxmax() if len(x) > 0 else np.nan

    clustered = (
        df_sorted
        .groupby(["VehicleNumber", "cluster_id"], as_index=False)
        .agg(
            EventTime=("EventTime", "min"),           # Startzeit des Clusters
            Duration_min=("Duration_min", "sum"),     # Gesamtdauer
            EventCategory=("EventCategory", most_frequent),
        )
    )

    clustered = clustered.drop(columns=["cluster_id"])
    return clustered


# ---------------------------------------------------------------------
# Feature Engineering: Zeitmerkmale & Rolling-Features
# ---------------------------------------------------------------------

def build_training_frame(events_df: pd.DataFrame):
    """
    Erzeugt Trainings-DataFrame auf Cluster-Level mit:
    - Zeit seit letzter Störung
    - Rolling-Features (6h, 24h)
    - zyklischen Zeitfeatures (Stunde, Wochentag)
    - Label: Störung in den nächsten HORIZON_H Stunden (hier: 1h)
    """
    df_sorted = events_df.sort_values(["VehicleNumber", "EventTime"]).copy()

    # Zeit bis zur nächsten Störung (für Label)
    df_sorted["Time_to_next"] = (
        df_sorted.groupby("VehicleNumber")["EventTime"].shift(-1)
        - df_sorted["EventTime"]
    )
    df_sorted["Time_to_next_hours"] = df_sorted["Time_to_next"].dt.total_seconds() / 3600

    # Zeit seit der letzten Störung (Feature)
    df_sorted["time_from_prev"] = (
        df_sorted["EventTime"]
        - df_sorted.groupby("VehicleNumber")["EventTime"].shift(1)
    )
    df_sorted["time_from_prev_h"] = df_sorted["time_from_prev"].dt.total_seconds() / 3600
    df_sorted["time_from_prev_h"] = df_sorted["time_from_prev_h"].fillna(0.0)
    df_sorted["time_from_prev_h_clipped"] = df_sorted["time_from_prev_h"].clip(upper=24.0)

    # Rolling-Features (6h / 24h) – EventTime als Index
    df_sorted = df_sorted.set_index("EventTime")

    roll_6_obj = df_sorted.groupby("VehicleNumber")["Duration_min"].rolling("6H")
    roll_6h = pd.DataFrame({
        "events_6h": roll_6_obj.count(),
        "downtime_6h": roll_6_obj.sum(),
    }).reset_index(level=0, drop=True)

    roll_24_obj = df_sorted.groupby("VehicleNumber")["Duration_min"].rolling("24H")
    roll_24h = pd.DataFrame({
        "events_24h": roll_24_obj.count(),
        "downtime_24h": roll_24_obj.sum(),
    }).reset_index(level=0, drop=True)

    df_sorted = df_sorted.join(roll_6h).join(roll_24h).reset_index()

    # Zeitbasierte Features
    hour_float = df_sorted["EventTime"].dt.hour + df_sorted["EventTime"].dt.minute / 60.0
    df_sorted["hour_float"] = hour_float
    df_sorted["weekday"] = df_sorted["EventTime"].dt.weekday  # 0=Montag

    # Zyklische Kodierung
    df_sorted["hour_sin"] = np.sin(2 * np.pi * df_sorted["hour_float"] / 24.0)
    df_sorted["hour_cos"] = np.cos(2 * np.pi * df_sorted["hour_float"] / 24.0)
    df_sorted["weekday_sin"] = np.sin(2 * np.pi * df_sorted["weekday"] / 7.0)
    df_sorted["weekday_cos"] = np.cos(2 * np.pi * df_sorted["weekday"] / 7.0)

    for col in ["events_6h", "downtime_6h", "events_24h", "downtime_24h"]:
        df_sorted[col] = df_sorted[col].fillna(0.0)

    # Label: Störung in den nächsten HORIZON_H Stunden?
    df_sorted["label_next1h"] = (df_sorted["Time_to_next_hours"] <= HORIZON_H).astype(int)

    # Letzte Ereignisse ohne folgendes Event entfernen
    model_df = df_sorted.dropna(subset=["Time_to_next_hours"]).copy()

    feature_cols = [
        "time_from_prev_h_clipped",
        "events_6h", "downtime_6h",
        "events_24h", "downtime_24h",
        "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
    ]

    return model_df, feature_cols


# ---------------------------------------------------------------------
# Modelltraining: globales Modell + Modelle pro Fahrzeug
# ---------------------------------------------------------------------

def train_models(model_df: pd.DataFrame, feature_cols: list):
    """
    Trainiert ein globales RandomForest-Modell und,
    falls genug Daten vorhanden sind, separate Modelle pro Fahrzeug.
    """
    X_global = model_df[feature_cols]
    y_global = model_df["label_next1h"]

    # Zeitlicher Split (70% Training, 30% "Test")
    split_time = model_df["EventTime"].quantile(0.7)
    train_mask = model_df["EventTime"] <= split_time
    X_train, y_train = X_global[train_mask], y_global[train_mask]

    global_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    global_model.fit(X_train, y_train)

    models_by_vehicle = {}
    stats = []

    for veh, df_v in model_df.groupby("VehicleNumber"):
        X_v = df_v[feature_cols]
        y_v = df_v["label_next1h"]

        n_samples = len(df_v)
        n_pos = int(y_v.sum())
        pos_rate = float(y_v.mean())

        if n_samples >= MIN_SAMPLES_PER_VEH and n_pos >= MIN_POS_PER_VEH:
            rf_v = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                class_weight="balanced",
            )
            rf_v.fit(X_v, y_v)
            models_by_vehicle[veh] = rf_v

        stats.append(
            {"VehicleNumber": veh, "n_samples": n_samples, "n_pos": n_pos, "pos_rate": pos_rate}
        )

    stats_df = pd.DataFrame(stats).sort_values("n_samples", ascending=False)
    return global_model, models_by_vehicle, stats_df


# ---------------------------------------------------------------------
# Risikoberechnung zum gemeinsamen Referenzzeitpunkt
# ---------------------------------------------------------------------

def compute_risk_for_reference_time(
    events: pd.DataFrame,
    models_by_vehicle: dict,
    global_model,
    feature_cols: list,
    ref_time,
) -> pd.DataFrame:
    """
    Berechnet für alle Fahrzeuge den Risikoscore 'Störung in der nächsten 1h'
    zu einem gemeinsamen Referenzzeitpunkt ref_time.

    Zusätzlich werden die 3 häufigsten EventCategory in den letzten 24h
    pro Fahrzeug als "wahrscheinlichste Ursachen" ausgegeben.
    """
    ref_time = pd.to_datetime(ref_time)

    vehicles = sorted(events["VehicleNumber"].unique().tolist())
    rows = []

    for veh in vehicles:
        df_v = events[events["VehicleNumber"] == veh]
        df_v = df_v[df_v["EventTime"] <= ref_time]

        if df_v.empty:
            # Annahme: lange kein Event -> wenig akute Aktivität
            time_from_prev_h = 24.0
            events_6h = 0
            downtime_6h = 0.0
            events_24h = 0
            downtime_24h = 0.0
            top3_causes = []
        else:
            last_event_time = df_v["EventTime"].max()
            delta_h = (ref_time - last_event_time).total_seconds() / 3600
            delta_h = max(delta_h, 0.0)
            time_from_prev_h = min(delta_h, 24.0)

            window_6h_start = ref_time - pd.Timedelta(hours=6)
            window_24h_start = ref_time - pd.Timedelta(hours=24)

            df_6h = df_v[df_v["EventTime"] > window_6h_start]
            df_24h = df_v[df_v["EventTime"] > window_24h_start]

            events_6h = len(df_6h)
            downtime_6h = df_6h["Duration_min"].sum()
            events_24h = len(df_24h)
            downtime_24h = df_24h["Duration_min"].sum()

            if not df_24h.empty:
                cause_counts = (
                    df_24h.groupby("EventCategory")
                          .size()
                          .sort_values(ascending=False)
                )
                top3_causes = cause_counts.head(3).index.tolist()
            else:
                cause_counts = (
                    df_v.groupby("EventCategory")
                        .size()
                        .sort_values(ascending=False)
                )
                top3_causes = cause_counts.head(3).index.tolist()

        hour_float_ref = ref_time.hour + ref_time.minute / 60.0
        weekday_ref = ref_time.weekday()

        hour_sin = np.sin(2 * np.pi * hour_float_ref / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_float_ref / 24.0)
        weekday_sin = np.sin(2 * np.pi * weekday_ref / 7.0)
        weekday_cos = np.cos(2 * np.pi * weekday_ref / 7.0)

        row = {
            "VehicleNumber": veh,
            "ref_time": ref_time,
            "time_from_prev_h_clipped": time_from_prev_h,
            "events_6h": events_6h,
            "downtime_6h": downtime_6h,
            "events_24h": events_24h,
            "downtime_24h": downtime_24h,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "weekday_sin": weekday_sin,
            "weekday_cos": weekday_cos,
            "top3_causes_str": ", ".join(top3_causes) if top3_causes else "",
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    X_ref = feat_df[feature_cols]

    risks = []
    for i, row in feat_df.iterrows():
        veh = row["VehicleNumber"]
        x = X_ref.iloc[i:i+1]
        model = models_by_vehicle.get(veh, global_model)
        p = model.predict_proba(x)[:, 1][0]
        risks.append(p)

    feat_df["risk_next1h"] = risks
    return feat_df


def compute_vehicle_risk_series(
    model_df: pd.DataFrame,
    models_by_vehicle: dict,
    global_model,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Erzeugt für alle Fahrzeuge eine Zeitreihe der Risikoscores pro Ereignis
    (Risiko = Störung innerhalb der nächsten 1h, bewertet zum Ereigniszeitpunkt).
    """
    rows = []

    for veh, df_v in model_df.groupby("VehicleNumber"):
        df_v_sorted = df_v.sort_values("EventTime").copy()
        X_v = df_v_sorted[feature_cols]
        model = models_by_vehicle.get(veh, global_model)
        df_v_sorted["risk_next1h"] = model.predict_proba(X_v)[:, 1]
        rows.append(df_v_sorted[["VehicleNumber", "EventTime", "risk_next1h", "label_next1h"]])

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["VehicleNumber", "EventTime", "risk_next1h", "label_next1h"])

    return out


# ---------------------------------------------------------------------
# Daten für Historische Statistik
# ---------------------------------------------------------------------

def enrich_events_for_dashboard(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Datum, Stunde und Wochentag für die historische Statistik hinzu.
    """
    df = events_df.copy()
    df["Date"] = df["EventTime"].dt.date
    df["Hour"] = df["EventTime"].dt.hour
    df["Weekday"] = df["EventTime"].dt.day_name()
    return df


# ---------------------------------------------------------------------
# Pipeline: Rohdaten -> Events -> Modelle -> Risiken
# ---------------------------------------------------------------------

st.sidebar.title("Einstellungen")

lookback_days = st.sidebar.slider(
    "Zeitraum für Daten aus der DB (Tage)",
    min_value=1,
    max_value=90,
    value=30,
)

with st.spinner("Daten aus der DB laden..."):
    df_raw = fetch_raw_data(lookback_days=lookback_days)

if df_raw.empty:
    st.error("Keine Daten aus der Datenbank im gewählten Zeitraum gefunden.")
    st.stop()

with st.spinner("Events vorbereiten, clustern und Features berechnen..."):
    events_df_raw = load_and_prepare_events(df_raw)
    events_df_raw = add_event_category(events_df_raw)
    events_df = cluster_events(events_df_raw, CLUSTER_THRESHOLD_MIN)
    model_df, feature_cols = build_training_frame(events_df)
    global_model, models_by_vehicle, stats_by_vehicle = train_models(model_df, feature_cols)
    risk_series_df = compute_vehicle_risk_series(model_df, models_by_vehicle, global_model, feature_cols)
    events_for_dash = enrich_events_for_dashboard(events_df)

# Referenzzeitpunkt: jetzt (kann auf der Prognose-Seite noch angepasst werden)
default_ref_time = pd.Timestamp.now()

# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------

page = st.sidebar.radio(
    "Seite wählen",
    ("Historische Statistik", "Prognose – alle Fahrzeuge", "Prognose – einzelnes Fahrzeug"),
)


# ---------------------------------------------------------------------
# Seite 1 – Historische Statistik
# ---------------------------------------------------------------------
if page == "Historische Statistik":
    st.title("Historische Statistik der AGV-Störungen")

    all_vehicles = sorted(events_for_dash["VehicleNumber"].unique().tolist())
    selected_vehicles = st.multiselect(
        "Fahrzeuge auswählen (leer = alle)",
        all_vehicles,
        default=[],
    )

    if selected_vehicles:
        df_filt = events_for_dash[events_for_dash["VehicleNumber"].isin(selected_vehicles)].copy()
    else:
        df_filt = events_for_dash.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anzahl Störungen", len(df_filt))
    with col2:
        st.metric("Anzahl Fahrzeuge", df_filt["VehicleNumber"].nunique())
    with col3:
        total_downtime_h = df_filt["Duration_min"].sum() / 60
        st.metric("Gesamtausfallzeit (h)", f"{total_downtime_h:.1f}")

    # Störungen pro Tag (mit Nullen)
    st.subheader("Störungen pro Tag")
    df_filt["Date_dt"] = pd.to_datetime(df_filt["Date"])
    events_per_day = df_filt.groupby("Date_dt").size()

    if not events_per_day.empty:
        full_idx = pd.date_range(
            start=events_per_day.index.min(),
            end=events_per_day.index.max(),
            freq="D",
        )
        events_per_day = events_per_day.reindex(full_idx, fill_value=0)
        events_per_day.index.name = "Date"
    st.bar_chart(events_per_day)

    # Gesamtausfallzeit pro Tag
    st.subheader("Gesamtausfallzeit pro Tag [Minuten]")
    downtime_per_day = df_filt.groupby("Date_dt")["Duration_min"].sum()
    if not downtime_per_day.empty:
        full_idx = pd.date_range(
            start=downtime_per_day.index.min(),
            end=downtime_per_day.index.max(),
            freq="D",
        )
        downtime_per_day = downtime_per_day.reindex(full_idx, fill_value=0)
        downtime_per_day.index.name = "Date"
    st.bar_chart(downtime_per_day)

    # Top-Fahrzeuge
    st.subheader("Top-Fahrzeuge nach Störungen und Ausfallzeit")
    kpi_vehicle = df_filt.groupby("VehicleNumber").agg(
        events=("EventTime", "count"),
        total_downtime_min=("Duration_min", "sum"),
        mean_downtime_min=("Duration_min", "mean"),
        max_downtime_min=("Duration_min", "max"),
    ).reset_index()

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Top 10 Fahrzeuge nach Anzahl Störungen")
        st.dataframe(
            kpi_vehicle.sort_values("events", ascending=False).head(10),
            use_container_width=True,
        )
    with col_b:
        st.write("Top 10 Fahrzeuge nach Gesamtausfallzeit")
        st.dataframe(
            kpi_vehicle.sort_values("total_downtime_min", ascending=False).head(10),
            use_container_width=True,
        )

    # Störungen nach Stunde
    st.subheader("Störungen nach Stunde des Tages")
    events_per_hour = df_filt.groupby("Hour").size()
    st.bar_chart(events_per_hour)

    # Störungen nach Wochentag
    st.subheader("Störungen nach Wochentag")
    weekday_order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_labels_de = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    events_per_weekday = df_filt.groupby("Weekday").size()
    events_per_weekday = events_per_weekday.reindex(weekday_order_en)
    events_per_weekday.index = weekday_labels_de
    st.bar_chart(events_per_weekday)

    st.caption(
        "Der Filter wirkt auf alle Kennzahlen oben. "
        "Bei leerer Auswahl werden alle Fahrzeuge berücksichtigt."
    )


# ---------------------------------------------------------------------
# Seite 2 – Prognose: alle Fahrzeuge (live)
# ---------------------------------------------------------------------
elif page == "Prognose – alle Fahrzeuge":
    st.title("Prognose – Risikoscore für alle Fahrzeuge (nächste Stunde, live)")

    ref_time = st.datetime_input(
        "Referenzzeitpunkt (Standard: jetzt)",
        value=default_ref_time,
    )

    events_for_risk = events_df[["VehicleNumber", "EventTime", "Duration_min", "EventCategory"]].copy()

    with st.spinner("Risiken pro Fahrzeug berechnen..."):
        risk_latest_df = compute_risk_for_reference_time(
            events=events_for_risk,
            models_by_vehicle=models_by_vehicle,
            global_model=global_model,
            feature_cols=feature_cols,
            ref_time=ref_time,
        )

    risk_latest_df["Risk (%)"] = (risk_latest_df["risk_next1h"] * 100).round(1)

    st.subheader(f"Risiko pro Fahrzeug (Referenzzeitpunkt: {pd.to_datetime(ref_time):%d.%m.%Y %H:%M})")
    st.dataframe(
        risk_latest_df[[
            "VehicleNumber",
            "Risk (%)",
            "top3_causes_str",
            "events_6h", "downtime_6h",
            "events_24h", "downtime_24h",
        ]],
        use_container_width=True,
    )

    st.subheader("Balkendiagramm – Risikoscore pro Fahrzeug (nächste Stunde)")
    st.bar_chart(
        data=risk_latest_df.set_index("VehicleNumber")["risk_next1h"],
        use_container_width=True,
    )

    st.caption(
        "Die Risikowerte basieren auf einem RandomForest-Modell, "
        "trainiert auf den Ereignissen der letzten Tage. "
        "Die Top-3-Ursachen je Fahrzeug stammen aus den häufigsten EventCategory "
        "in den letzten 24 Stunden."
    )


# ---------------------------------------------------------------------
# Seite 3 – Prognose: einzelnes Fahrzeug (live)
# ---------------------------------------------------------------------
elif page == "Prognose – einzelnes Fahrzeug":
    st.title("Prognose – einzelnes Fahrzeug (nächste Stunde, live)")

    vehicles = sorted(events_df["VehicleNumber"].unique().tolist())
    selected_vehicle = st.selectbox("Fahrzeug wählen", vehicles)

    ref_time = st.datetime_input(
        "Referenzzeitpunkt (Standard: jetzt)",
        value=default_ref_time,
        key="ref_time_single",
    )

    events_for_risk = events_df[["VehicleNumber", "EventTime", "Duration_min", "EventCategory"]].copy()

    with st.spinner("Risiko für alle Fahrzeuge berechnen..."):
        risk_latest_df = compute_risk_for_reference_time(
            events=events_for_risk,
            models_by_vehicle=models_by_vehicle,
            global_model=global_model,
            feature_cols=feature_cols,
            ref_time=ref_time,
        )

    veh_row = risk_latest_df[risk_latest_df["VehicleNumber"] == selected_vehicle]

    if veh_row.empty:
        st.info("Für dieses Fahrzeug liegen im gewählten Zeitraum keine Events vor.")
    else:
        risk_value = float(veh_row["risk_next1h"].iloc[0])
        risk_percent = risk_value * 100
        causes_str = veh_row["top3_causes_str"].iloc[0]

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Risikoverteilung (Pie-Chart, nächste Stunde)")

            fig, ax = plt.subplots()
            labels = [
                "Störung innerhalb der nächsten Stunde",
                "Keine Störung innerhalb der nächsten Stunde",
            ]
            sizes = [risk_value, 1 - risk_value]

            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.markdown(f"**Aktueller Risikowert:** {risk_percent:.1f}%")
            st.markdown(f"**Wahrscheinlichste Ursachen (Top 3):** {causes_str or '–'}")

        with col_right:
            st.subheader("Historische Störungen des Fahrzeugs")

            veh_events = events_for_dash[events_for_dash["VehicleNumber"] == selected_vehicle].copy()
            if not veh_events.empty:
                veh_events = veh_events.sort_values("EventTime")
                st.line_chart(
                    data=veh_events.set_index("EventTime")[["Duration_min"]],
                    use_container_width=True,
                )
                st.dataframe(
                    veh_events[["EventTime", "Duration_min", "EventCategory"]],
                    use_container_width=True,
                )
            else:
                st.info("Keine historischen Events im gewählten Zeitraum.")

        st.caption(
            "Der Risikowert basiert auf einem Modell mit Vorhersagehorizont von 1 Stunde. "
            "Die Zeitreihe zeigt die historisch beobachteten Störungen des Fahrzeugs."
        )
