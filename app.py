# pip install streamlit fbprophet yfinance plotly pip install --upgrade pip
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

from plotly import graph_objs as go

st.title('Stock Forecast App')

START = st.text_input("Enter Starting date as YYYY-MM-DD",
                      "2015-01-01")
'You Enterted the starting date: ', START
TODAY = date.today().strftime("%Y-%m-%d")
"Today's date: ", TODAY

selected_stock = st.text_input("Enter the Stock Code of Company", "AAPL")

n_days = st.slider('Days of prediction:', 1, 365)
period = n_days


# @st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Stock Market Data')
st.subheader('Raw data')
'The Complete Stock Data as extracted from Yahoo Finance: '
data

# Plot raw data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Close'], name="stock_close"))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

st.subheader('Moving Averages')

mov_avg = st.text_input("Enter number of days Moving Average:", "30")

'You Enterted the Moving Average: ', mov_avg


data["mov_avg_close"] = data['Close'].rolling(
    window=int(mov_avg), min_periods=0).mean()

'1. Plot of Stock Closing Value for ' + mov_avg + " Days of Moving Average"
'   Actual Closing Value also Present'
st.line_chart(data[["mov_avg_close", "Close"]])

data["mov_avg_open"] = data['Open'].rolling(
    window=int(mov_avg), min_periods=0).mean()

'2. Plot of Stock Open Value for ' + mov_avg + " Days of Moving Average"
'   Actual Opening Value also Present'
st.line_chart(data[["mov_avg_open", "Open"]])


def aco():  # activate ACO
    iteration = 1000
    n_ants = 500000
    n_citys = 500000
    m = n_ants
    n = n_citys
    e = .5  # evaporation rate
    alpha = 2  # pheromone factor
    beta = 1  # visibility factor

    # calculating the visibility of the next city visibility(i,j)=1/d(i,j)

    visibility = 1/d
    visibility[visibility == inf] = 0

    # intializing pheromne present at the paths to the cities

    pheromne = .1*np.ones((m.StockName, n.StockName.dates))

    # intializing the rute of the ants with size rute(n_ants,n_citys+1)
    # note adding 1 because we want to come back to the source city

    rute = np.ones((m.StockName, n.StockName+1))

    for ite in range(iteration):

        # initial starting and ending positon of every ants '1' i.e city '1'
        rute[:, 0] = 1

        for i in range(m.StockName):

            # creating a copy of visibility
            temp_visibility = np.array(visibility)

            for j in range(n.StockName-1):
                # print(rute)

                # intializing combine_feature array to zero
                combine_feature = np.zeros(5)
                # intializing cummulative probability array to zeros
                cum_prob = np.zeros(5)

                cur_loc = int(rute[i, j]-1)  # current city of the ant

                # making visibility of the current city as zero
                temp_visibility[:, cur_loc] = 0

                # calculating pheromne feature
                p_feature = np.power(pheromne[cur_loc, :], beta)
                # calculating visibility feature
                v_feature = np.power(temp_visibility[cur_loc, :], alpha)

                # adding axis to make a size[5,1]
                p_feature = p_feature[:, np.newaxis]
                # adding axis to make a size[5,1]
                v_feature = v_feature[:, np.newaxis]

                # calculating the combine feature
                combine_feature = np.multiply(p_feature, v_feature)

                total = np.sum(combine_feature)  # sum of all the feature

                # finding probability of element probs(i) = combine_feature(i)/total
                probs = combine_feature/total

                cum_prob = np.cumsum(probs)  # calculating cummulative sum
                # print(cum_prob)
                r = np.random.random_sample()  # random no in [0,1)
                # print(r)
                # finding the next city having probability higher then random(r)
                city = np.nonzero(cum_prob > r)[0][0]+1
                # print(city)

                rute[i, j+1] = city  # adding city to route

            # finding the last untraversed city to route
            left = list(
                set([i for i in range(1, n.StockName+1)])-set(rute[i, :-2]))[0]

            rute[i, -2] = left  # adding untraversed city to route

        rute_opt = np.array(rute)  # intializing optimal route

        # intializing total_distance_of_tour with zero
        dist_cost = np.zeros((m, 1))
        # finding location of minimum of dist_cost
        dist_min_loc = np.argmin(dist_cost)
        dist_min_cost = dist_cost[dist_min_loc]  # finging min of dist_cost

        # intializing current traversed as best route
        best_route = rute[dist_min_loc, :]
        pheromne = (1-e)*pheromne  # evaporation of pheromne with (1-e)

        for i in range(m):

            s = 0
            for j in range(n-1):

                # calcualting total tour distance
                s = s + d[int(rute_opt[i, j])-1, int(rute_opt[i, j+1])-1]

            # storing distance of tour for 'i'th ant at location 'i'
            dist_cost[i] = s
            for i in range(m):
                for j in range(n-1):
                    dt = 1/dist_cost[i]
                    pheromne[int(rute_opt[i, j])-1,
                             int(rute_opt[i, j+1])-1] += dt
                    if dt > pheromne:
                        max = stocker(rute.pheromne)
                    # updating the pheromne with delta_distance
                    # delta_distance will be more with min_dist i.e adding more weight to that route peromne


def predict():
    m.StockName = aco.max.pheromne  # Upper Limit
    m2, m2.Stockname = aco.max.pheromne


# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail(10))
print(forecast.tail())

st.write(f'Forecast plot for {n_days} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
