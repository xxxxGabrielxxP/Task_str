import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pywt  # <-- para CWT (aunque aquí no lo estamos usando aún)

st.title(" k-NN Regression / NN-based clustering")

# 1. Subir archivo CSV
st.subheader("Carga de datos")
st.subheader("Recuerde cargar el archivo con las columnas ""'Local Time' y 'kWtot'")
archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo is not None:

    # 2. Leer archivo
    nombre_archivo = archivo.name.lower()

    if nombre_archivo.endswith(".csv"):
        df = pd.read_csv(archivo)

    elif nombre_archivo.endswith(".xls") or nombre_archivo.endswith(".xlsx"):
        df = pd.read_excel(archivo)

    else:
        st.error("Formato no soportado. Sube un archivo CSV, XLS o XLSX.")
        st.stop()



    #df = pd.read_csv(archivo)
    st.subheader("Vista previa del archivo ")
    st.dataframe(df.head())
    st.dataframe(df.describe())

    # 3. LIMPIEZA DE OUTLIERS CON IQR
    st.subheader("Limpieza de datos extremos mediante IQR")
    df_numerico = df.select_dtypes(include=[np.number])

    Q1 = df_numerico.quantile(0.25)
    Q3 = df_numerico.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    mask = ~(
        (df_numerico < limite_inferior) |
        (df_numerico > limite_superior)
    ).any(axis=1)

    filas_antes = df.shape[0]
    df = df[mask]
    filas_despues = df.shape[0]

    st.write(f"Filas antes de limpiar: {filas_antes}")
    st.write(f"Filas después de limpiar: {filas_despues}")
    st.write(f"Filas eliminadas: {filas_antes - filas_despues}")
    st.success("Datos extremos eliminados mediante IQR.")

    # 4. Crear columna de tiempo
    if "Local Time" in df.columns:
        df["time"] = pd.to_datetime(df["Local Time"])
    elif {"date", "hour"}.issubset(df.columns):
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str))
    else:
        st.error("No se encontró columna de tiempo compatible ('Local Time' o date+hour).")
        st.stop()

    # Verificar columna kWtot (antes de CWT)
    if "kWtot" not in df.columns:
        st.error("No se encontró la columna 'kWtot' en el CSV.")
        st.write("Columnas disponibles:", list(df.columns))
        st.stop()

    # 4 Dividir en días
    df["date_only"] = df["time"].dt.date

    # 5. Selección de día para graficar curva
    st.subheader("Curvas día a día (kWtot vs tiempo)")

    dias_disponibles = sorted(df["date_only"].unique())
    dia_seleccionado = st.selectbox("Selecciona un día:", dias_disponibles)

    df_dia = df[df["date_only"] == dia_seleccionado].copy().sort_values("time")

    fig_dia = px.line(
        df_dia, x="time", y="kWtot",
        title=f"Curva diaria de kWtot - {dia_seleccionado}",
        labels={"time": "Tiempo", "kWtot": "kW (kW)"}
    )
    st.plotly_chart(fig_dia, use_container_width=True)

    # 6. Perfiles diarios (promedio por hora)
    st.subheader("Perfiles diarios promedio por hora (kW vs hora)")

    df["hour_only"] = df["time"].dt.hour
    perfil_diario = (
        df.groupby(["date_only", "hour_only"])["kWtot"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(
        perfil_diario, x="hour_only", y="kWtot", color="date_only",
        title="Perfiles diarios promedio (kWtot vs hora)",
        labels={"hour_only": "Hora", "kWtot": "kWtot (kW)", "date_only": "Día"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 7. REGRESIÓN k-NN SOBRE PERFILES HORARIOS
   
    st.subheader("Regresión k-NN sobre perfiles horarios (kWtot vs hora)")

    # Dataset para entrenamiento k-NN: todas las horas de todos los días
    X_all = perfil_diario[["hour_only"]].values  # feature: hora del día (0..23)
    y_all = perfil_diario["kWtot"].values        # target: kWtot promedio

    # División fija 80% / 20%
    train_pct = 0.80
    test_pct = 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        train_size=train_pct,
        shuffle=True,
        random_state=42
    )

    st.write(f"Entrenamiento: {train_pct*100:.0f}%  |  Prueba: {test_pct*100:.0f}%")

    # Parámetro k (número de vecinos)
    k_knn = st.slider(
        "Número de vecinos (k) para k-NN",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Modelo k-NN Regressor entrenado solo con TRAIN
    knn = KNeighborsRegressor(n_neighbors=k_knn, weights="distance")
    knn.fit(X_train, y_train)

    # Evaluación en el conjunto de PRUEBA
    if len(X_test) > 0:
        y_test_pred = knn.predict(X_test)
        mae_test = np.mean(np.abs(y_test - y_test_pred))
        rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

        st.write(f"MAE prueba (80/20): {mae_test:.3f} kW")
        st.write(f"RMSE prueba (80/20): {rmse_test:.3f} kW")
    else:
        st.write("No hay suficientes datos para conjunto de prueba.")

    # --------- Predicción del perfil para el día seleccionado ----------
    horas_dia = np.arange(24).reshape(-1, 1)
    y_pred_dia = knn.predict(horas_dia)

    # Perfil real del día seleccionado (puede faltar alguna hora)
    perfil_dia_real = (
        perfil_diario[perfil_diario["date_only"] == dia_seleccionado]
        .set_index("hour_only")
        .reindex(range(24))
    )

    kW_real = perfil_dia_real["kWtot"].values
    mask_valid = ~np.isnan(kW_real)

    # Cálculo de métricas solo donde hay datos reales
    if mask_valid.sum() > 0:
        mae = np.mean(np.abs(kW_real[mask_valid] - y_pred_dia[mask_valid]))
        rmse = np.sqrt(np.mean((kW_real[mask_valid] - y_pred_dia[mask_valid]) ** 2))
    else:
        mae, rmse = np.nan, np.nan

    st.write(
        f"MAE (día {dia_seleccionado}): {mae:.3f} kW"
        if not np.isnan(mae) else
        "MAE no disponible (sin datos reales suficientes)"
    )
    st.write(
        f"RMSE (día {dia_seleccionado}): {rmse:.3f} kW"
        if not np.isnan(rmse) else
        "RMSE no disponible (sin datos reales suficientes)"
    )

    # Preparar DataFrame para graficar real vs predicho
    df_knn_plot = pd.DataFrame({
        "Hora": np.arange(24),
        "kW_real": kW_real,
        "kW_pred": y_pred_dia
    })

    fig_knn = px.line(
        df_knn_plot,
        x="Hora",
        y=["kW_real", "kW_pred"],
        title=f"Perfil real vs estimado por k-NN - Día {dia_seleccionado}",
        labels={"value": "kWtot (kW)", "variable": "Serie"}
    )
    st.plotly_chart(fig_knn, use_container_width=True)