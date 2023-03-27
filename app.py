import pandas as pd
import streamlit as st
from PIL import Image
from bokeh.models.widgets import Div
import plotly.express as px
import nltk
import graphviz
import base64


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
    df_raw = pd.read_csv(r'data/internet_service_churn.csv')
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
        js = "window.location.href = 'https://muarrikhyazka.github.io'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    if st.button('🍱 GITHUB'):
        js = "window.location.href = 'https://github.com/muarrikhyazka'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)








st.title(title)


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

st.subheader('Data Understanding')
st.write(
    """
    **Source : Take it from yfinance API, you can see on the notebook how I gotten it**
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
    Open :
    Close : 
    """
)

st.subheader('Method')
st.write(
    """
    Used LSTM Neural Network
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



st.subheader('Insights')
st.write(
    """
    I calculated word frequency and see on top 10 in unigram and bigram. Try to see all chart combination between sentiment and category and will show you which has insight.
    """
)

## convert to corpus
top=10
corpus = df["content_clean"][(df['sentiment']=='NEGATIVE') & (df['predicted_category']=='INTERFACE')]
lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))

    
## calculate words bigrams
dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
dtf_bi = pd.DataFrame(dic_words_freq.most_common(), 
                      columns=["Word","Freq"])
dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(
                   string for string in x) )
fig_bi = px.bar(dtf_bi.iloc[:top,:].sort_values(by="Freq"), x="Freq", y="Word", orientation='h',
             hover_data=["Word", "Freq"],
             height=400,
             title='Top 10 Bigrams Text')
st.plotly_chart(fig_bi, use_container_width=True)

st.write("""
    We can see here, its combination between negative sentiment and interface category. 
    It shows us that interface in TV is needed to be improved because android tv was mentioned sometimes. 
    Many other words which is related to TV such as mi Box (Xiaomi set top box for TV), dolby digital (sound in smart tv), and nvidia shield (android tv-based digital media player).
    It indicates that Netflix should prioritize to improve their app in TV. 
    Furthermore, in detail many complaints for voice search feature, so It should be attention to start.
""")

c1, c2 = st.columns(2)
with c1:
    st.info('**[Github Repo](https://github.com/muarrikhyazka/internet-service-provider-churn-prediction)**', icon="🍣")

