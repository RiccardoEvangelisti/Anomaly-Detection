import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("nba.csv")
print("shapes:", df.shape)

sns.kdeplot(df["PTS"])
sns.boxplot(df["PTS"])
plt.show()

df_X = pd.DataFrame(df[["MIN", "3PM", "3PA", "FTM", "FTA"]])
df_Y = pd.DataFrame(df["PTS"])
X_train, X_test, Y_train, Y_test = train_test_split(
    df_X, df_Y, test_size=0.2, random_state=1
)

# min max scaler
scaler = MinMaxScaler()
scaler.fit(X_train)  # trova min e max
X_train = scaler.transform(X_train)  # applica la formula del min-max al train
X_test = scaler.transform(X_test)  # applica la formula del min-max al test
# ora sia le colonne del train sia del test hanno valori "scalati" (tra zero e uno)
# attenzione che i dati del test possono non essere tra zero e uno perché lo scaler lo si calcola solo sul train ovviamente

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
estate_y_pred = regr.predict(X_test)

# Show the beta coefficients
print("Beta_0: \n", regr.coef_)
print("Beta_1: \n", regr.intercept_)
# Compute the RSS
mse = mean_squared_error(Y_test, estate_y_pred)
print("Mean Square Error:", mse)
# Compute the R-square index
rsquare = r2_score(Y_test, estate_y_pred)
print("R-square:", rsquare)

print(
    regr.coef_
)  # sono la cosa più importante da stampare, ossia il coefficiente più alto si riferisce alla colonna più rilevante nella predizione

# stampa ...
beta0 = regr.intercept_[0]
beta1 = regr.coef_[0][0]

# plt.scatter(X_test[0], Y_test, color="green")
# plt.plot(X_test, estate_y_pred, color="red", linewidth=3)
# plt.show()
