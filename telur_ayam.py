import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from streamlit_folium import folium_static
#from folium_datetime import DateFilter
import base64
from io import BytesIO
#sns.set(style='dark')

df = pd.read_excel('df_fix.xlsx')

datetime_columns = ["Tanggal"]
df.sort_values(by="Tanggal", inplace=True)
df.reset_index(inplace=True)
 
for column in datetime_columns:
    df[column] = pd.to_datetime(df[column])

min_date = df["Tanggal"].min()
max_date = df["Tanggal"].max()

#Header 
st.set_page_config(layout='wide')
st.markdown(
    """
    <h1 style="text-align: center;">Informasi Harga Telur Ayam Ras di Banyumas</h1>
    """,
    unsafe_allow_html=True,
)

st.subheader("     ")
st.subheader("     ")

st.markdown("<h2 style='text-align: center;'>Peta Pasar di Banyumas</h2>", unsafe_allow_html=True)

# Fitur Rentang Waktu
with st.container():
    #st.subheader("Rentang Waktu")
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

df1 = df[(df["Tanggal"] >= str(start_date)) & 
                (df["Tanggal"] <= str(end_date))]
df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])

#Create HTML for images
image_stream = BytesIO()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df1['Tanggal'], df1['Pasar Manis'], color='red')
ax.set_title('Harga di Pasar Manis', fontsize=14)
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga')
ax.legend()
plt.savefig(image_stream, format='png')
plt.close()
image_base64 = base64.b64encode(image_stream.getvalue()).decode("utf-8")
html = f'<img src="data:image/png;base64,{image_base64}">'

image_stream1 = BytesIO()
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(df1['Tanggal'], df1['Pasar Wage'], color = 'blue')
ax1.set_title('Harga di Pasar Wage', fontsize=14)
ax1.set_xlabel('Tanggal')
ax1.set_ylabel('Harga')
ax1.legend()
plt.savefig(image_stream1, format='png')
plt.close()

image_base641 = base64.b64encode(image_stream1.getvalue()).decode("utf-8")
html1 = f'<img src="data:image/png;base64,{image_base641}">'

# Create Folium map
location1 = [-7.417745006891739, 109.22726059533683] #koordinat pasar manis
m = folium.Map(location=location1, zoom_start=13)
popup_m = folium.Popup(html, max_width=1000)
folium.Marker(location1, popup=popup_m).add_to(m)

location2 = [-7.426639020453093, 109.24882526626837]#koordinat pasar wage
n = folium.Map(location = location2, zoom_start=13)
popup_n = folium.Popup(html1, max_width=1000)
folium.Marker(location2, popup=popup_n).add_to(n)

# Add marker and popup for Pasar Manis (m)
popup_m = folium.Popup(html, max_width=1000)
folium.Marker([-7.417745006891739, 109.22726059533683], popup=popup_m).add_to(m)

# Add marker and popup for Pasar Wage (n)
popup_n = folium.Popup(html1, max_width=1000)
folium.Marker([-7.426639020453093, 109.24882526626837], popup=popup_n).add_to(m)

# Display the Streamlit map with markers
folium_static(m, width=1150, height = 600)
st.subheader("     ")

st.markdown("<h2 style='text-align: center;'>Grafik Harga Telur Ayam Ras</h2>", unsafe_allow_html=True)
    
#Create the graphs for Pasar Manis and Pasar Wage
selected_market = st.selectbox(
    "Pilih Pasar:",
    ("Pasar Manis", "Pasar Wage", "Kedua Pasar"),
)
    
# Tampilkan grafik berdasarkan pasar yang dipilih
if selected_market == 'Pasar Manis':
    #st.write("### Harga Telur Ayam Ras di Pasar Manis")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df1['Tanggal'], df1['Pasar Manis'], color='red', label='Pasar Manis')
    ax.set_title('Harga di Pasar Manis', fontsize=12)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
        
elif selected_market == 'Pasar Wage':
    #st.write("### Harga Telur Ayam Ras di Pasar Wage")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df1['Tanggal'], df1['Pasar Wage'], color='blue', label='Pasar Wage')
    ax.set_title('Harga di Pasar Wage', fontsize=12)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
else:
    #st.write("### Perbandingan Harga Telur Ayam Ras di Kedua Pasar")
    fig, ax2=plt.subplots(figsize=(8, 3))

    ax2.plot(df1['Tanggal'], df1['Pasar Manis'], color='red', label='Pasar Manis')
    ax2.plot(df1['Tanggal'], df1['Pasar Wage'], color='blue', label='Pasar Wage')
    ax2.set_title('Harga di Pasar Wage dan Pasar Manis', fontsize=12)
    ax2.set_xlabel('Tanggal')
    ax2.set_ylabel('Harga')  
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig)
st.subheader("     ")

#Tabel
st.markdown("<h2 style='text-align: center;'>Detail Harga Telur Ayam Ras</h2>", unsafe_allow_html=True)
#Create the graphs for Pasar Manis and Pasar Wage
selected_market = st.selectbox(
    "Pilih Pasar:",
    ("Pasar Manis", "Pasar Wage", "Kedua Pasar"),key="selecbox_market"
)
# Tampilkan detail harga berdasarkan pasar yang dipilih
if selected_market == 'Pasar Manis':
    st.dataframe(df1[['Tanggal', 'Pasar Manis']], width=1900)
        
elif selected_market == 'Pasar Wage':
    st.dataframe(df1[['Tanggal',  'Pasar Wage']], width=1900)
else:
    st.dataframe(df1[['Tanggal', 'Pasar Manis', 'Pasar Wage']], width=1900)