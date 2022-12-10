import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ApiGetData
import ta
from ArimaModel import ArimaModel
from io import StringIO
import sys
import plotly.express as px

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
