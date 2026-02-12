import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="AGV-Störungen – Dashboard", layout="wide")

# Pfade zu den vorab generierten Dateien (aus modell.ipynb)
EVENTS_FILE = Path("model_outputs/events_for_dashboard.csv")
RISK_LATEST_FILE = Path("model_outputs/risk_latest.csv")
RISK_SERIES_FILE = Path("model_outputs/risk_series.csv")


# -----------------------------------------------------------
# Daten laden
# -----------------------------------------------------------

@st.cache_data
def load_events():
    df = pd.read_csv(EVENTS_FILE, parse_dates=["EventTime"])
    # Es wird angenommen, dass die Datei bereits folgende Spalten enthält:
    # Date, Hour, Weekday, VehicleNumber, Duration_min, EventCategory
    return df


@st.cache_data
def load_risk_latest():
    # Erwartete Spalten:
    # VehicleNumber, risk_next1h, top3_causes_str,
    # events_6h, downtime_6h, events_24h, downtime_24h, ref_time (optional)
    df = pd.read_csv(RISK_LATEST_FILE)
    return df


@st.cache_data
def load_risk_series():
    # Erwartete Spalten:
    # VehicleNumber, EventTime, risk_next1h, label_next1h
    df = pd.read_csv(RISK_SERIES_FILE, parse_dates=["EventTime"])
    return df


events_df = load_events()
risk_latest_df = load_risk_latest()
risk_series_df = load_risk_series()


# -----------------------------------------------------------
# Navigation
# -----------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite wählen",
    ("Historische Statistik", "Prognose – alle Fahrzeuge", "Prognose – einzelnes Fahrzeug"),
)


# -----------------------------------------------------------
# Seite 1 – Historische Statistik
# -----------------------------------------------------------
if page == "Historische Statistik":
    st.title("Historische Statistik der AGV-Störungen")

    all_vehicles = sorted(events_df["VehicleNumber"].unique().tolist())
    selected_vehicles = st.multiselect(
        "Fahrzeuge auswählen (leer = alle)",
        all_vehicles,
        default=[]
    )

    if selected_vehicles:
        df_filt = events_df[events_df["VehicleNumber"].isin(selected_vehicles)].copy()
    else:
        df_filt = events_df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anzahl Störungen", len(df_filt))
    with col2:
        st.metric("Anzahl Fahrzeuge", df_filt["VehicleNumber"].nunique())
    with col3:
        total_downtime_h = df_filt["Duration_min"].sum() / 60
        st.metric("Gesamtausfallzeit (h)", f"{total_downtime_h:.1f}")

    # --- Störungen pro Tag (mit Nullen für Tage ohne Störungen) ---
    st.subheader("Störungen pro Tag")

    # Sicherstellen, dass Date als Datetime interpretiert wird (nicht nur als Date-Objekt)
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

    # --- Gesamtausfallzeit pro Tag [Minuten] (ebenfalls mit Nullen) ---
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

    # --- Rest wie gehabt: Top-Fahrzeuge, Stunde, Wochentag ---
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

    st.subheader("Störungen nach Stunde des Tages")
    events_per_hour = df_filt.groupby("Hour").size()
    st.bar_chart(events_per_hour)

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


# -----------------------------------------------------------
# Seite 2 – Prognose: alle Fahrzeuge (aus risk_latest.csv)
# -----------------------------------------------------------
elif page == "Prognose – alle Fahrzeuge":
    st.title("Prognose – Risikoscore für alle Fahrzeuge (nächste Stunde)")

    st.markdown(
        "Die unten stehenden Risikowerte wurden im Modell-Notebook berechnet und "
        "als Datei gespeichert. Das Dashboard zeigt **nur die Ergebnisse**."
    )

    # Falls in risk_latest_df eine Spalte ref_time vorhanden ist, den Referenzzeitpunkt anzeigen
    ref_time_str = ""
    if "ref_time" in risk_latest_df.columns:
        try:
            ref_time_parsed = pd.to_datetime(risk_latest_df["ref_time"].iloc[0])
            ref_time_str = f" (Referenzzeitpunkt: {ref_time_parsed:%d.%m.%Y %H:%M})"
        except Exception:
            ref_time_str = ""

    risk_tbl = risk_latest_df.copy()
    # Erwartete Spalte: risk_next1h
    risk_tbl["Risk (%)"] = (risk_tbl["risk_next1h"] * 100).round(1)

    st.subheader(f"Risiko pro Fahrzeug{ref_time_str}")
    st.dataframe(
        risk_tbl[["VehicleNumber", "Risk (%)", "top3_causes_str",
                  "events_6h", "downtime_6h", "events_24h", "downtime_24h"]],
        use_container_width=True,
    )

    st.subheader("Balkendiagramm – Risikoscore pro Fahrzeug (nächste Stunde)")
    st.bar_chart(
        data=risk_tbl.set_index("VehicleNumber")["risk_next1h"],
        use_container_width=True,
    )

    st.caption(
        "Die Top-3-Ursachen je Fahrzeug stammen aus der Modell-Auswertung "
        "(häufigste EventCategory in einem definierten Zeitfenster)."
    )


# -----------------------------------------------------------
# Seite 3 – Prognose: einzelnes Fahrzeug (aus risk_series.csv)
# -----------------------------------------------------------
elif page == "Prognose – einzelnes Fahrzeug":
    st.title("Prognose – einzelnes Fahrzeug (nächste Stunde)")

    vehicles = sorted(risk_series_df["VehicleNumber"].unique().tolist())
    selected_vehicle = st.selectbox("Fahrzeug wählen", vehicles)

    veh_series = risk_series_df[risk_series_df["VehicleNumber"] == selected_vehicle].copy()

    if veh_series.empty:
        st.info("Für dieses Fahrzeug liegen keine Modellvorhersagen in der Datei vor.")
    else:
        st.markdown(
            "Die unten dargestellten Risikowerte stammen direkt aus der Datei "
            "`risk_series.csv`, die im Modell-Notebook erzeugt wurde "
            "(Vorhersagehorizont: nächste Stunde)."
        )

        # Letzten verfügbaren Risikowert als „aktuellen“ Wert interpretieren
        last_row = veh_series.sort_values("EventTime").iloc[-1]
        risk_value = float(last_row["risk_next1h"])
        risk_percent = risk_value * 100

        # Falls risk_latest_df vorhanden ist: Top-3-Ursachen dort auslesen
        causes_str = ""
        row_latest = risk_latest_df[risk_latest_df["VehicleNumber"] == selected_vehicle]
        if not row_latest.empty and "top3_causes_str" in row_latest.columns:
            causes_str = row_latest["top3_causes_str"].iloc[0]

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
            if causes_str:
                st.markdown(f"**Wahrscheinlichste Ursachen (Top 3):** {causes_str}")
            else:
                st.markdown("**Wahrscheinlichste Ursachen (Top 3):** –")

        with col_right:
            st.subheader("Historischer Verlauf des Risikos (ereignisbasiert, nächste Stunde)")

            veh_series_plot = veh_series.sort_values("EventTime").set_index("EventTime")
            st.line_chart(
                data=veh_series_plot[["risk_next1h"]],
                use_container_width=True,
            )

            tbl = veh_series[["EventTime", "risk_next1h", "label_next1h"]].copy()
            tbl["Risk (%)"] = (tbl["risk_next1h"] * 100).round(1)
            tbl.rename(
                columns={"label_next1h": "Tatsächliche Störung innerhalb 1h (Label)"},
                inplace=True,
            )

            st.markdown("Detailtabelle der Ereignisse und Modellvorhersagen:")
            st.dataframe(
                tbl[["EventTime", "Risk (%)", "Tatsächliche Störung innerhalb 1h (Label)"]],
                use_container_width=True,
            )

        st.caption(
            "Die Linie zeigt die vom Modell berechneten Risikowerte "
            "zum jeweiligen Ereigniszeitpunkt (Vorhersagehorizont: 1 Stunde)."
        )
