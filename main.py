import matplotlib.pyplot as plt
import pandas

from sklearn.linear_model import LinearRegression

data = [
    {
        "year1": {
            1: 4.8,
            2: 4.1,
            3: 6.0,
            4: 1.5
        }
    },
    {
        "year2": {
            1: 5.8,
            2: 5.2,
            3: 6.8,
            4: 7.4
        }
    },
    {
        "year3": {
            1: 6.0,
            2: 5.6,
            3: 7.5,
            4: 7.8
        }
    },
    {
        "year4": {
            1: 9.3,
            2: 5.9,
            3: 8.0,
            4: 8.4
        }
    }
]

if __name__ == '__main__':
    years = []
    quarters = []
    values = []
    for item in data:
        for value in item.values():
            for point_key in value.keys():
                years.append(list(item.keys())[0])
                quarters.append(point_key)
                values.append(value.get(point_key))

    dataFrame = pandas.DataFrame({
        'id': range(len(values)),
        'year': years,
        'quarter': quarters,
        'value': values})

    dataFrame['MA'] = dataFrame['value'].rolling(window=4).mean()
    dataFrame['CMA'] = dataFrame['MA'].rolling(window=2).mean()
    dataFrame['Yt/CMA'] = dataFrame['value'] / dataFrame['CMA']

    seasonal_index = dict()

    for quarter in quarters:
        seasonal_index[quarter] = dataFrame.loc[dataFrame['quarter'] == quarter]['Yt/CMA'].mean()

    dataFrame['st'] = dataFrame['quarter'].map(seasonal_index)
    dataFrame['normalized_value'] = dataFrame['value'] / dataFrame['st']

    xAxis = pandas.DataFrame(dataFrame['id'])
    yAxis = pandas.DataFrame(dataFrame['normalized_value'])

    model = LinearRegression()
    model.fit(xAxis, yAxis)
    linearR_intercept = model.intercept_[0]
    linearR_coef = model.coef_[0][0]
    print('intercept:', linearR_intercept)
    print('coefficient:', linearR_coef)

    dataFrame['trend'] = dataFrame['id'].apply(lambda x: x * linearR_coef + linearR_intercept)

    datapoint = dataFrame['value'].values.tolist()
    trend = dataFrame['trend'].values.tolist()

    print(dataFrame)

    forecast_df = pandas.DataFrame({
        'id': range(len(values) + 8),

    })
    forecast_df['quarter'] = forecast_df['id'].apply(lambda x: x % 4 + 1)
    forecast_df['st'] = forecast_df['quarter'].map(seasonal_index)
    forecast_df['trend'] = forecast_df['id'].apply(lambda x: (x + 1) * linearR_coef + linearR_intercept)
    forecast_df['forecast'] = forecast_df['trend'] * forecast_df['st']
    # forecast_df['poly_forecast'] = model2.predict(pandas.DataFrame(forecast_df['id']))

    print(forecast_df)
    predicted = forecast_df['forecast'].values.tolist()

    fig, ax = plt.subplots()

    ax.plot(datapoint, marker=".", )
    ax.plot(trend, marker=".")
    ax.plot(predicted, marker="o")

    plt.plot(yAxis)

    plt.legend(('Data', 'trend', 'prediction', 'normalized_data'),
               loc='lower right')

    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='r', labelsize='medium', width=1)

    plt.show()
