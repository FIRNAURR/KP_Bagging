import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model, save_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# -------------------------
# Fungsi encode logo ke base64
# -------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -------------------------
# Load logo
# -------------------------
try:
    logo_base64 = get_base64_of_bin_file("logo.png")
except:
    logo_base64 = None

# -------------------------
# Tambah CSS Pos Indonesia
# -------------------------
st.markdown(
    f"""
    <style>
    body {{
        background-color: #ffffff;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }}

    .pos-logo {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;
        margin-top: 10px;
        margin-bottom: 10px;
    }}

    .title-pos {{
        text-align: center;
        color: #F37021;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 0px;
    }}

    .subtitle-pos {{
        text-align: center;
        color: #1C3F94;
        font-size: 18px;
        margin-top: -5px;
        margin-bottom: 30px;
    }}

    .stButton > button {{
        background-color: #F37021;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5em 2em;
        font-weight: bold;
    }}

    .stButton > button:hover {{
        background-color: #1C3F94;
        color: white;
    }}

    .stAlert {{
        border-left: 5px solid #F37021;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header + Logo
# -----------------------------
if logo_base64:
    st.markdown(
        f"""
        <img class="pos-logo" src="data:image/png;base64,{logo_base64}">
        <div class="title-pos">Prediksi Waktu Bagging Berdasarakan Faktor Oprasional</div>
        <div class="title-pos">Kantor Pos KCU Jember</div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class="title-pos">Prediksi Waktu Bagging Volume Kiriman</div>
        <div class="subtitle-pos">Kantor Pos KCU Jember</div>
        """,
        unsafe_allow_html=True
    )
# -----------------------------
# Load Model
# -----------------------------
try:
    model = load_model("lstm_model.keras", compile=False)
    model.compile(optimizer='adam', loss='mae')
    st.success("✅ Model berhasil diload dan dikompilasi ulang.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Load Scaler
# -----------------------------
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("✅ Scaler berhasil diload.")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.stop()

# -----------------------------
# Upload Data
# -----------------------------
st.subheader("📥 Upload Data Excel (Opsional)")

st.info("""
- Pastikan file dalam format Excel (.xlsx atau .xls).
- Data harus sudah diakumulasi per hari (misalnya 1 baris = 1 hari).
- Kolom minimal:
    - Volume Kiriman
    - Berat Kiriman
    - Jumlah Staf
""")

uploaded_file = st.file_uploader(
    "Upload file Excel untuk prediksi batch (opsional):",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df_upload = pd.read_excel(uploaded_file)
        st.success("✅ File berhasil di-upload dan dibaca.")
        st.write("Data yang di-upload:")
        st.dataframe(df_upload, use_container_width=True)

        # Mapping kolom (case-insensitive)
        colmap = {c.lower().strip(): c for c in df_upload.columns}

        required_cols = ["volume kiriman", "jumlah staf"]

        # Cari nama kolom Berat Kiriman
        berat_col = None
        if "berat kiriman" in colmap:
            berat_col = colmap["berat kiriman"]
        elif "berat kiriman (kg)" in colmap:
            berat_col = colmap["berat kiriman (kg)"]
        else:
            st.error("❌ Kolom 'Berat Kiriman' atau 'Berat Kiriman (Kg)' tidak ditemukan di file.")
            st.stop()

        for col in required_cols:
            if col not in colmap:
                st.error(f"❌ Kolom '{col}' tidak ditemukan di file.")
                st.stop()

        all_predictions = []

        for idx, row in df_upload.iterrows():
            volume = row[colmap["volume kiriman"]]
            berat = row[berat_col]
            staf_asli = row[colmap["jumlah staf"]]

            waktu_per_staf = {}
            previous_waktu = None

            # Prediksi waktu untuk staf 2-5
            for staf in range(2, 6):
                dummy_bagging = 0.0
                row_input = [volume, berat, staf, dummy_bagging]

                timesteps = 30
                x_input_array = np.tile(row_input, (timesteps, 1))
                x_input_scaled = scaler.transform(x_input_array)
                x_input = x_input_scaled.reshape(1, timesteps, 4)

                y_pred_scaled = model.predict(x_input, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                waktu_pred = max(y_pred[0][3], 0)

                if previous_waktu is not None and waktu_pred >= previous_waktu:
                    waktu_pred = previous_waktu * 0.9

                waktu_per_staf[staf] = waktu_pred
                previous_waktu = waktu_pred

            # Cari staf tercepat
            batas_kerja = 600
            batas_waktu_bagging = 540

            staf_memenuhi = {s: w for s, w in waktu_per_staf.items() if w <= batas_waktu_bagging}

            if staf_memenuhi:
                staf_tercepat = min(staf_memenuhi.keys())
                waktu_tercepat = staf_memenuhi[staf_tercepat]
            else:
                staf_tercepat = min(waktu_per_staf, key=waktu_per_staf.get)
                waktu_tercepat = waktu_per_staf[staf_tercepat]

            all_predictions.append({
                "Index": idx,
                "Volume Kiriman": volume,
                "Berat Kiriman": berat,
                "Jumlah Staf (Asli)": staf_asli,
                "Prediksi Waktu Bagging (menit)": waktu_per_staf.get(staf_asli, None),
                "Rekomendasi Jumlah Staf": staf_tercepat,
                "Prediksi Waktu Sesuai Rekomendasi (menit)": waktu_tercepat
            })

            # -----------------------------
            # Tampilkan grafik per baris
            # -----------------------------
            st.subheader(f"📊 Grafik Prediksi Baris Data ke-{idx+1}")

            df_compare = pd.DataFrame({
                "Jumlah Staf": [f"{s} Staf" for s in waktu_per_staf.keys()],
                "Waktu Bagging (menit)": list(waktu_per_staf.values())
            })

            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(
                data=df_compare,
                x="Jumlah Staf",
                y="Waktu Bagging (menit)",
                color="#1C3F94",
                ax=ax
            )

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f'{height:.0f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black'
                )

            ax.set_title("Prediksi Waktu Bagging per Jumlah Staf", fontsize=12, color="#F37021")
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            st.pyplot(fig)

            # -----------------------------
            # Tampilkan Kesimpulan per baris
            # -----------------------------
            st.markdown("---")
            st.subheader("📝 Kesimpulan")

            if waktu_tercepat <= batas_waktu_bagging:
                st.success(
                    f"✅ Rekomendasi: Gunakan {int(staf_tercepat)} staf "
                    f"karena waktu bagging diprediksi hanya {waktu_tercepat:.2f} menit. "
                    f"Ini masih di bawah batas {batas_waktu_bagging} menit, "
                    f"sehingga operasional dinilai aman dalam jam kerja {batas_kerja} menit/hari."
                )
            else:
                st.warning(
                    f"⚠ Hasil prediksi menunjukkan waktu bagging {waktu_tercepat:.2f} menit "
                    f"dengan {int(staf_tercepat)} staf, yang melebihi batas {batas_waktu_bagging} menit. "
                    f"Pertimbangkan menambah staf, membagi shift, atau melakukan optimasi proses "
                    f"agar bagging selesai di bawah {batas_waktu_bagging} menit per hari."
                )

        # -----------------------------
        # Tampilkan hasil keseluruhan
        # -----------------------------
        df_result = pd.DataFrame(all_predictions)

        st.subheader("📊 Hasil Prediksi Batch Upload")
        st.dataframe(df_result, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi error saat membaca atau memproses file: {e}")

# -----------------------------
# Form Input Manual
# -----------------------------
st.subheader("✍ Atau Masukkan Data Secara Manual")

col1, col2 = st.columns(2)

with col1:
    volume = st.number_input("Volume Kiriman (paket):", value=100000, step=10)
with col2:
    berat = st.number_input("Berat Kiriman (kg):", value=100000, step=10)

if st.button("Prediksi Manual dan Rekomendasi Staf"):

    prediksi_waktu_per_staf = {}
    previous_waktu = None

    for staf in range(2, 6):
        dummy_bagging = 0.0
        row = [volume, berat, staf, dummy_bagging]

        timesteps = 30
        x_input_array = np.tile(row, (timesteps, 1))
        x_input_scaled = scaler.transform(x_input_array)
        x_input = x_input_scaled.reshape(1, timesteps, 4)

        y_pred_scaled = model.predict(x_input, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)

        waktu_pred = max(y_pred[0][3], 0)

        if previous_waktu is not None and waktu_pred >= previous_waktu:
            waktu_pred = previous_waktu * 0.9

        prediksi_waktu_per_staf[staf] = waktu_pred
        previous_waktu = waktu_pred

    batas_kerja = 600
    batas_waktu_bagging = 540

    staf_memenuhi = {s: w for s, w in prediksi_waktu_per_staf.items() if w <= batas_waktu_bagging}

    if staf_memenuhi:
        staf_tercepat = min(staf_memenuhi.keys())
        waktu_tercepat = prediksi_waktu_per_staf[staf_tercepat]
    else:
        staf_tercepat = min(prediksi_waktu_per_staf, key=prediksi_waktu_per_staf.get)
        waktu_tercepat = prediksi_waktu_per_staf[staf_tercepat]

    st.success("✅ Hasil Perbandingan Prediksi Waktu Bagging (2-5 Staf):")

    for staf, waktu in prediksi_waktu_per_staf.items():
        st.write(f"- {staf} Staf → {waktu:.2f} menit")

    df_compare = pd.DataFrame({
        "Jumlah Staf": [f"{s} Staf" for s in prediksi_waktu_per_staf.keys()],
        "Waktu Bagging (menit)": list(prediksi_waktu_per_staf.values())
    })

    st.subheader("📊 Grafik Batang Input Manual")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        data=df_compare,
        x="Jumlah Staf",
        y="Waktu Bagging (menit)",
        color="#1C3F94",
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:.0f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

    ax.set_title("Perbandingan Waktu Bagging 2-5 Staf", fontsize=12, color="#F37021")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    st.dataframe(df_compare, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Kesimpulan")

    if waktu_tercepat <= batas_waktu_bagging:
        st.success(
            f"✅ Rekomendasi: Gunakan {staf_tercepat} staf "
            f"karena waktu bagging diprediksi hanya {waktu_tercepat:.2f} menit. "
            f"Ini masih di bawah batas {batas_waktu_bagging} menit, "
            f"sehingga operasional dinilai aman dalam jam kerja {batas_kerja} menit/hari."
        )
    else:
        st.warning(
            f"⚠ Hasil prediksi menunjukkan waktu bagging {waktu_tercepat:.2f} menit "
            f"dengan {staf_tercepat} staf, yang melebihi batas {batas_waktu_bagging} menit. "
            f"Pertimbangkan menambah staf, membagi shift, atau melakukan optimasi proses "
            f"agar bagging selesai di bawah {batas_waktu_bagging} menit per hari."
        )
