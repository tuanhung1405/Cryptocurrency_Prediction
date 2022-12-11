import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ApiGetData
import ta
from ArimaModel import ArimaModel
from io import StringIO
import sys
import plotly.express as px
from PIL import Image

st.set_page_config(page_title='Cryptocurrency Price Prediction',page_icon="📶",layout="wide")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://www.uel.edu.vn" target="_blank">
  <img class="image-25"src="https://www.uel.edu.vn/Resources/Images/SubDomain/HomePage/Style/logo_uel.png" width="350"
  </a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

h = st.markdown("""
<style>
div.fullScreenFrame > div {
    display: flex;
    justify-content: center;
}
</style>""", unsafe_allow_html=True)

#Title
original_title = '<p style="text-align: center; color:#3498DB; text-shadow: 2px 2px 4px #000000; font-size: 60px;">Dự Đoán Khả Năng Chịu Đựng Rủi Ro</p>'
st.markdown(original_title, unsafe_allow_html=True)

st.write("""Trang web này sử dụng mô hình Machine Learning để dự đoán điểm ** Khả năng chịu rủi ro **!
Dữ liệu thu được của *** Nhóm sinh viên UEL *** được lấy từ một cuộc khảo sát với hơn 500 người tham gia tại Thành phố Hồ Chí Minh.""")

#col1, col2, col3 = st.columns(3)
#with col1:
 #   st.write(' ')
#with col2:
#    image = Image.open('NGHIÊN CỨU KHOA HỌC (1).png')
 #   st.image(image, caption='Members of Group 35')
#with col3:
 #  st.write('   ')
  
background = Image.open("Nguyễn Tuấn Hưng_ Ảnh chân dung.png")
col1, col2, col3 = st.columns([0.2, 1, 0.2])
col2.image(background, use_column_width=True)
    
st.write('---')
tup, coinname = ApiGetData.getListCoins()


def main():

    st.title("CRYPTOCURRENCY PRICE PREDICTION USING ARIMA MODEL")
    st.subheader("This is project of Nguyễn Tuấn Hưng from UEL")
    st.text("Data are crawled from coinbase ")
    st.write("!!!! Enjoy and have a good day !!!!")

    st.sidebar.write("Choose your coin and the period")
    coins = st.sidebar.selectbox("Which coin", (tup))
    period = st.sidebar.selectbox("Choose the period", ("DAY", "1WEEK", "2WEEK", "MONTH"))

    name = "Coin: " + coinname.get(coins)
    st.subheader(name)
    data = ApiGetData.getFinalData(coins, period)
    st.dataframe(data)

    data["MA20"] = ta.trend.sma_indicator(data['close'], window=20)
    data["MA50"] = ta.trend.sma_indicator(data['close'], window=50)
    data["MA100"] = ta.trend.sma_indicator(data['close'], window=100)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['open'],
                                 high=data['high'],
                                 low=data['low'],
                                 close=data['close'], name="OHLC"),
                  row=1, col=1)
    fig.add_trace(go.Line(x=data.index, y=data['MA20'], name="MA20", line=dict(
        color="purple",
        width=1)))
    fig.add_trace(go.Line(x=data.index, y=data['MA50'], name="MA50", line=dict(
        color="yellow",
        width=1.5)))
    fig.add_trace(go.Line(x=data.index, y=data['MA100'], name="MA100", line=dict(
        color="orange",
        width=2)))

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], showlegend=False), row=2, col=1)

    # Do not show OHLC's rangeslider plot
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.update_layout(
        autosize=False,
        width=780,
        height=540,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        )
    )

    st.plotly_chart(fig)

    model = ArimaModel(data, period)

    st.write("Now prepare for the prediction, note that the prediction below uses the Arima model as a reference. You should not apply it to your portfolio and the author will not bear any associated liability.")
    st.write("HAVE FUN and wish success")
    period = st.slider("Chose period you want to predict", 1, 5, 1)
    if st.button("START PREDICT"):
        st.warning(model.checkData())
        model.createDataReturn()
        st.write("Stationality test")
        warn, ADF, p_value = model.checkStationarity()
        s1 = "ADF Statistic: " + str(ADF)
        s2 = "p-value: " + str(p_value)
        st.text(s1)
        st.text(s2)
        st.warning(warn)

        st.markdown("**_Running the auto_arima can take a while. Please wait!!!_**")

        result = model.displaySummary()

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        print(result.summary())
        sys.stdout = old_stdout
        st.text(mystdout.getvalue())

        pre = model.predict(period)
        st.write("The data predict")
        st.dataframe(pre)

        fig2 = px.line(data, y="close", x=data.index)
        fig2.add_trace(
            go.Scatter(x=pre.index, y=pre['Price_mean'], line=dict(color="red"), name="forecast"))
        fig2.add_trace(go.Scatter(x=pre.index, y=pre['Price_upper'], line=dict(color="green", dash='dash'), name="upper", ))
        fig2.add_trace(go.Scatter(x=pre.index, y=pre['Price_lower'], line=dict(color="green", dash='dash'), name="lower", ))
        st.plotly_chart(fig2)


if __name__ == '__main__':
    main()
