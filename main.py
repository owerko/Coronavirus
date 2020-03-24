import pandas as pd
import matplotlib as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import numpy as np

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

df = pd.read_csv(url)
dfPL = df.iloc[183, 4:]
y = dfPL.tolist()


def createList(r1, r2):
    return [item for item in range(r1, r2 + 1)]


x = createList(1, len(y))
print(x)
print(y)


def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))


fit = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])
errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]
print(fit)
print(errors)
