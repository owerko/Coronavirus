import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import numpy as np

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

df = pd.read_csv(url)
dfPL = df.iloc[183, 4:]
y = dfPL.tolist()
#print(dfPL)

def createList(r1, r2):
    return [item for item in range(r1, r2 + 1)]


x = createList(1, len(y))
print(x)
print(len(x))
print(y)
print(len(y))

def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))


fit = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])
errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]
a = fit[0][0]
b = fit[0][1]
c = fit[0][2]
#print(fit[0])
#print(errors)

sol = int(fsolve(lambda x: logistic_model(x, a, b, c) - int(c), b))

print(f'Szacowana liczna zarażony wyniesie {fit[0][2]:.0f} +- {errors[2]:.0f}')
print(f'Zatrzymanie epidemii nzastąpi po {sol} dniach od 22 stycznia 2020')


def exponential_model(x, a, b, c):
    return a * np.exp(b * (x - c))


exp_fit = curve_fit(exponential_model, x, y, p0=[1, 1, 1])

#print(exp_fit[0])

pred_x = list(range(max(x), sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x, y, label="Real data", color="red")

plt.plot(x + pred_x, [logistic_model(i, fit[0][0], fit[0][1], fit[0][2]) for i in x + pred_x], label="Logistic model")
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 22 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()
