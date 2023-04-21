import pandas as pd
import streamlit as st
from PIL import Image
from bokeh.models.widgets import Div
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import graphviz
import base64
from statsmodels.compat.numpy import NP_LT_123
from pandas.util._decorators import Appender
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm




title = 'Time Series Forecasting of Spotify Stock Price'




# Layout
img = Image.open('assets/icon_pink-01.png')
st.set_page_config(page_title=title, page_icon=img, layout='wide')






st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
#   width: 50%;
}
</style> """, unsafe_allow_html=True)


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

file_name='style.css'
with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)






# Content
@st.cache
def load_data():
    df_raw = pd.read_csv(r'data/spotify_stock_price.csv', sep=';')
    df = df_raw.copy()
    return df_raw, df

df_raw, df = load_data()



def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" class="center" width="100" height="100"/>' % b64
    st.write(html, unsafe_allow_html=True)


# Sidebar color
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ef4da0;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    f = open("assets/icon-01.svg","r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)

    st.write('\n')
    st.write('\n')
    st.write('\n')

    if st.button('🏠 HOME'):
        # js = "window.location.href = 'http://www.muarrikhyazka.com'"  # Current tab
        js = "window.open('http://www.muarrikhyazka.com')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    if st.button('🍱 GITHUB'):
        # js = "window.location.href = 'https://www.github.com/muarrikhyazka'"  # Current tab
        js = "window.open('https://www.github.com/muarrikhyazka')"
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)








st.title(title)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Business Understanding')
st.write(
    """
    Stock price change of company are very fast. We can get high profit if we can buy and sell on the best timing.
    Stock price forecasting is very important tools to get high profit so we can know when is the best time to buy and sell.
    """
)

st.write(
    """
    In this case, I take Spotify or in code SPOT. There is no specific scientific reason why choose it, I just love the company and the product. Hope someday will be there.
    """
)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Data Understanding')
st.write(
    """
    **Source : Take it from yfinance API, you can see on the notebook how I gotten it.**
    """
)

st.write(
    """
    **Below is sample of the data.** 
    """
)

st.dataframe(df.sample(5))

st.write(
    """
    **Below is my understanding about each column**
    \nOpen : stock price when the opening bell rang or the stock market open.
    \nHigh : highest stock price during one period (on our case, its one day).
    \nLow : lowest stock price during one period (on our case, its one day).
    \nClose : stock price when the stock exchange closed shop for the day.
    \nAdj Close : the closing price after adjustments for all applicable splits and dividend distributions.
    \nVolume : the number of shares traded in a particular stock over a specific period of time.
    """
)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Method')
st.write(
    """
    Used LSTM Neural Network.
    """
)

st.write("""
    **Flowchart**
""")

graph = graphviz.Digraph()
graph.edge('EDA', 'Assumption Test')
graph.edge('Assumption Test', 'Data Preprocessing')
graph.edge('Data Preprocessing', 'Modeling')
graph.edge('Modeling', 'Evaluation')



st.graphviz_chart(graph)

st.write(
    """
    \n
    \n
    \n
    """
)

st.subheader('Modeling')
st.write(
    """
    Before do the modeling, I need to do some tests and on it.
    """
)

st.write(
    """
    **Plot**
    """
)
st.write(
    """
    Want to see the trend.
    """
)

## preprocessing
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']

## build plotting function
def plot_line(data, cols =[]):
    for col in cols:
        plt.figure(figsize=(20, 5))
        plt.plot(data[col])
        plt.grid(color='black')
        plt.title(col)
        plt.xticks(rotation=90)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.show()
        st.pyplot(ax.figure)

plot_line(df,['Close'])

st.write(
    """
    **PACF (Partial Auto Correlation Function)**
    """
)
st.write(
    """
    Want to check is it correlated with past data or not and how far.
    """
)

pacf = sm.graphics.tsa.plot_pacf(df['Close'], lags=26, method="ywm")
st.pyplot(pacf.figure)

st.write(
    """
    From this plot, it looks like a statistically significant correlation may exist until two years.
    """
)

st.write(
    """
    **Seasonal Decomposition**
    """
)
st.write(
    """
    Want to check is there any seasonal pattern in the data.
    """
)

decom = seasonal_decompose(df[['Close']], model='additive', period=12).plot()
st.pyplot(decom.figure)

st.write(
    """
    From the plot, we can see there is seasonal pattern.
    """
)

st.write(
    """
    **Stationarity Check**
    """
)
st.write(
    """
    Want to check whether stationary or not.
    """
)

## stationarity check
result = adfuller(df['Close'])
st.code(f'Test Statistics: {result[0]}')
st.code(f'p-value: {result[1]}')
st.code(f'critical_values: {result[4]}')

if result[1] > 0.05:
    st.code("Series is not stationary")
else:
    st.code("Series is stationary")

st.write(
    """
    **Modeling**
    """
)
st.write(
    """
    Lets move to modeling. There are some step which I didnt show, they are : 
    \n1. Split the data.
    \n2. Determine parameter.
    \n3. Fit the model.
    """
)

st.write(
    """
    Below is the comparison between train, validation, and prediction data. The model performance on RMSE is :
    """
)

st.code('5.08830294388938')

## load data in modeling
@st.cache
def load_data_modeling(allow_output_mutation=True):
    train_df = pd.read_csv(r'data/train.csv', sep=';')
    valid_df = pd.read_csv(r'data/valid.csv', sep=';')
    return train_df, valid_df

train_raw, valid_raw = load_data_modeling()

train= train_raw.copy()
valid= valid_raw.copy()


train['Date'] = pd.to_datetime(train['Date'])
train.index = train['Date']

valid['Date'] = pd.to_datetime(valid['Date'])
valid.index = valid['Date']


## visualize the data
result = plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
st.pyplot(result.figure)

st.write("""
    With RMSE around 5, this model is good enough to be implemented on production.
""")

st.write(
    """
    \n
    \n
    \n
    """
)

c1, c2 = st.columns(2)
with c1:
    st.info('**[Github Repo](https://github.com/muarrikhyazka/time-series-forecasting-of-stock-price)**', icon="🍣")

