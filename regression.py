import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

def linreg (x_train, x_test, y_train):
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    print(reg.intercept_, reg.coef_)
    return y_pred

def KNN(x_train, x_test, y_train, y_test, n):
    reg = KNeighborsRegressor(n_neighbors=n)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred

def dec_tree(x_train, x_test, y_train, y_test):
    reg = DecisionTreeRegressor(min_samples_leaf=7, min_samples_split=7, criterion='squared_error')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred

def RF(x_train, x_test, y_train, y_test):
    reg = RandomForestRegressor(min_samples_leaf=7, min_samples_split=7, criterion='squared_error')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_pred

def verif (y_pred, y_test, name):
    print('___________', name, '____________')
    print('r2 %s'%(r2_score(y_pred, y_test)))
    print('MAPE %s'%(mean_absolute_percentage_error(y_pred, y_test)))

def transform (df): #делаем колонки числовыми
    for column in df.columns:
        if (df[column].dtype != 'float64') & (df[column].dtype != 'int64'):
            enc = LabelEncoder()
            enc.fit(df[column])
            df[column] = enc.transform(df[column])
    return df

df = pd.read_csv(r'C:\Users\Ирина\Desktop\учеба\питон\diamonds.csv')
df = transform(df)

#нет привязки к оси времени, не временной ряд, необходимо удалить пропуски
#для кодирования - заменяем строку на целое число
#удаляем ненужные столбцы
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.info()
print(df.isnull().sum())

#линейная регрессия применяется при построении функции, применяем непрерывные переменные
#построим корреляцию Пирсона (линейные зависимости)
#корреляция есть, если коэффициент больше 0,5 по модулю
#если близко к -1, то линейная зависимость обратная, если близко к 1, то линейная зависимость прямая
new_df = df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']]
sns.heatmap(new_df.corr().round(2), annot = True, cbar = False)
plt.show()

new_df.plot.box(title = 'Ящик с усами')
plt.show()

#проверка нормального распределения
#построим гистограмму
plt.hist(df['price'])
plt.show()
sns.displot(df['price'], kde = True)
plt.show()
sns.scatterplot(x = df['carat'], y = df['price'])
plt.show()

print(df.describe())

describe = df.describe()

#3 стандартных отклонения
lower_limit = df['price'].mean() - 3 * df['price'].std() #левая граница
upper_limit = df['price'].mean() + 3 * df['price'].std() #правая граница
df = df[(df['price'] >= lower_limit) | (df['price'] <= upper_limit)]
df = df.dropna()

x_train, x_test, y_train, y_test = train_test_split(df[['carat']], df[df.columns[-1]],
                                                     random_state=3)
verif(linreg (x_train, x_test, y_train), y_test, 'Linear regression')
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]], df[df.columns[-1]],
                                                     random_state=2)
verif(KNN(x_train, x_test, y_train, y_test, 5), y_test, 'KNN')
verif(dec_tree(x_train, x_test, y_train, y_test), y_test, 'Decision tree')
verif(RF(x_train, x_test, y_train, y_test), y_test, 'Random forest')
