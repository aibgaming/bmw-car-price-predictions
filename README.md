# BMW Car Price Predictions

```python
import pandas as pd
import numpy as np
```

# Data Exploration


```python
data = pd.read_csv("bmw.csv")
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>year</th>
      <th>price</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5 Series</td>
      <td>2014</td>
      <td>11200</td>
      <td>Automatic</td>
      <td>67068</td>
      <td>Diesel</td>
      <td>125</td>
      <td>57.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6 Series</td>
      <td>2018</td>
      <td>27000</td>
      <td>Automatic</td>
      <td>14827</td>
      <td>Petrol</td>
      <td>145</td>
      <td>42.8</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5 Series</td>
      <td>2016</td>
      <td>16000</td>
      <td>Automatic</td>
      <td>62794</td>
      <td>Diesel</td>
      <td>160</td>
      <td>51.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1 Series</td>
      <td>2017</td>
      <td>12750</td>
      <td>Automatic</td>
      <td>26676</td>
      <td>Diesel</td>
      <td>145</td>
      <td>72.4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7 Series</td>
      <td>2014</td>
      <td>14500</td>
      <td>Automatic</td>
      <td>39554</td>
      <td>Diesel</td>
      <td>160</td>
      <td>50.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10776</th>
      <td>X3</td>
      <td>2016</td>
      <td>19000</td>
      <td>Automatic</td>
      <td>40818</td>
      <td>Diesel</td>
      <td>150</td>
      <td>54.3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>10777</th>
      <td>5 Series</td>
      <td>2016</td>
      <td>14600</td>
      <td>Automatic</td>
      <td>42947</td>
      <td>Diesel</td>
      <td>125</td>
      <td>60.1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>10778</th>
      <td>3 Series</td>
      <td>2017</td>
      <td>13100</td>
      <td>Manual</td>
      <td>25468</td>
      <td>Petrol</td>
      <td>200</td>
      <td>42.8</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>10779</th>
      <td>1 Series</td>
      <td>2014</td>
      <td>9930</td>
      <td>Automatic</td>
      <td>45000</td>
      <td>Diesel</td>
      <td>30</td>
      <td>64.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>10780</th>
      <td>X1</td>
      <td>2017</td>
      <td>15981</td>
      <td>Automatic</td>
      <td>59432</td>
      <td>Diesel</td>
      <td>125</td>
      <td>57.6</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>10781 rows × 9 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10781 entries, 0 to 10780
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   model         10781 non-null  object 
     1   year          10781 non-null  int64  
     2   price         10781 non-null  int64  
     3   transmission  10781 non-null  object 
     4   mileage       10781 non-null  int64  
     5   fuelType      10781 non-null  object 
     6   tax           10781 non-null  int64  
     7   mpg           10781 non-null  float64
     8   engineSize    10781 non-null  float64
    dtypes: float64(2), int64(4), object(3)
    memory usage: 758.2+ KB
    

# Data Cleansing

All data is unique


```python
data.nunique()
```




    model             24
    year              25
    price           3777
    transmission       3
    mileage         8086
    fuelType           5
    tax               38
    mpg              102
    engineSize        17
    dtype: int64



No data is missing


```python
data.isna().sum()
```




    model           0
    year            0
    price           0
    transmission    0
    mileage         0
    fuelType        0
    tax             0
    mpg             0
    engineSize      0
    dtype: int64



# Target column


```python
# Seperating out the target column
X = data.drop(columns="price")
y = data["price"]
```

# Feature Engineering


```python
car_type = {'5 Series':'sedan',
 '6 Series':'coupe',
 '1 Series':'coupe',
 '7 Series':'sedan',
 '2 Series':'coupe',
 '4 Series':'coupe',
 'X3':'suv',
 '3 Series':'sedan',
 'X5':'suv',
 'X4':'suv',
 'i3':'electric',
 'X1':'suv',
 'M4':'sports',
 'X2':'suv',
 'X6':'suv',
 '8 Series':'coupe',
 'Z4':'convertible',
 'X7':'suv',
 'M5':'sports',
 'i8':'electric',
 'M2':'sports',
 'M3':'sports',
 'M6':'sports',
 'Z3':'convertible'}

# Feature Engineering
# We're going to add a classification that we will add based on our domain-specific knowledge 
X["model"] = X["model"].str.strip()
X["car_type"] = X["model"].map(car_type)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>year</th>
      <th>transmission</th>
      <th>mileage</th>
      <th>fuelType</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>car_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5 Series</td>
      <td>2014</td>
      <td>Automatic</td>
      <td>67068</td>
      <td>Diesel</td>
      <td>125</td>
      <td>57.6</td>
      <td>2.0</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6 Series</td>
      <td>2018</td>
      <td>Automatic</td>
      <td>14827</td>
      <td>Petrol</td>
      <td>145</td>
      <td>42.8</td>
      <td>2.0</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5 Series</td>
      <td>2016</td>
      <td>Automatic</td>
      <td>62794</td>
      <td>Diesel</td>
      <td>160</td>
      <td>51.4</td>
      <td>3.0</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1 Series</td>
      <td>2017</td>
      <td>Automatic</td>
      <td>26676</td>
      <td>Diesel</td>
      <td>145</td>
      <td>72.4</td>
      <td>1.5</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7 Series</td>
      <td>2014</td>
      <td>Automatic</td>
      <td>39554</td>
      <td>Diesel</td>
      <td>160</td>
      <td>50.4</td>
      <td>3.0</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10776</th>
      <td>X3</td>
      <td>2016</td>
      <td>Automatic</td>
      <td>40818</td>
      <td>Diesel</td>
      <td>150</td>
      <td>54.3</td>
      <td>2.0</td>
      <td>suv</td>
    </tr>
    <tr>
      <th>10777</th>
      <td>5 Series</td>
      <td>2016</td>
      <td>Automatic</td>
      <td>42947</td>
      <td>Diesel</td>
      <td>125</td>
      <td>60.1</td>
      <td>2.0</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>10778</th>
      <td>3 Series</td>
      <td>2017</td>
      <td>Manual</td>
      <td>25468</td>
      <td>Petrol</td>
      <td>200</td>
      <td>42.8</td>
      <td>2.0</td>
      <td>sedan</td>
    </tr>
    <tr>
      <th>10779</th>
      <td>1 Series</td>
      <td>2014</td>
      <td>Automatic</td>
      <td>45000</td>
      <td>Diesel</td>
      <td>30</td>
      <td>64.2</td>
      <td>2.0</td>
      <td>coupe</td>
    </tr>
    <tr>
      <th>10780</th>
      <td>X1</td>
      <td>2017</td>
      <td>Automatic</td>
      <td>59432</td>
      <td>Diesel</td>
      <td>125</td>
      <td>57.6</td>
      <td>2.0</td>
      <td>suv</td>
    </tr>
  </tbody>
</table>
<p>10781 rows × 9 columns</p>
</div>



# Data Encoding

Using one hot encoding with get_dummies()


```python
X.car_type.value_counts()
```




    coupe          4340
    sedan          3605
    suv            2451
    sports          210
    convertible     115
    electric         60
    Name: car_type, dtype: int64




```python
X = pd.get_dummies(X, drop_first=True)
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10781 entries, 0 to 10780
    Data columns (total 39 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   year                    10781 non-null  int64  
     1   mileage                 10781 non-null  int64  
     2   tax                     10781 non-null  int64  
     3   mpg                     10781 non-null  float64
     4   engineSize              10781 non-null  float64
     5   model_2 Series          10781 non-null  uint8  
     6   model_3 Series          10781 non-null  uint8  
     7   model_4 Series          10781 non-null  uint8  
     8   model_5 Series          10781 non-null  uint8  
     9   model_6 Series          10781 non-null  uint8  
     10  model_7 Series          10781 non-null  uint8  
     11  model_8 Series          10781 non-null  uint8  
     12  model_M2                10781 non-null  uint8  
     13  model_M3                10781 non-null  uint8  
     14  model_M4                10781 non-null  uint8  
     15  model_M5                10781 non-null  uint8  
     16  model_M6                10781 non-null  uint8  
     17  model_X1                10781 non-null  uint8  
     18  model_X2                10781 non-null  uint8  
     19  model_X3                10781 non-null  uint8  
     20  model_X4                10781 non-null  uint8  
     21  model_X5                10781 non-null  uint8  
     22  model_X6                10781 non-null  uint8  
     23  model_X7                10781 non-null  uint8  
     24  model_Z3                10781 non-null  uint8  
     25  model_Z4                10781 non-null  uint8  
     26  model_i3                10781 non-null  uint8  
     27  model_i8                10781 non-null  uint8  
     28  transmission_Manual     10781 non-null  uint8  
     29  transmission_Semi-Auto  10781 non-null  uint8  
     30  fuelType_Electric       10781 non-null  uint8  
     31  fuelType_Hybrid         10781 non-null  uint8  
     32  fuelType_Other          10781 non-null  uint8  
     33  fuelType_Petrol         10781 non-null  uint8  
     34  car_type_coupe          10781 non-null  uint8  
     35  car_type_electric       10781 non-null  uint8  
     36  car_type_sedan          10781 non-null  uint8  
     37  car_type_sports         10781 non-null  uint8  
     38  car_type_suv            10781 non-null  uint8  
    dtypes: float64(2), int64(3), uint8(34)
    memory usage: 779.2 KB
    

# Scaling the dataset


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>mileage</th>
      <th>tax</th>
      <th>mpg</th>
      <th>engineSize</th>
      <th>model_2 Series</th>
      <th>model_3 Series</th>
      <th>model_4 Series</th>
      <th>model_5 Series</th>
      <th>model_6 Series</th>
      <th>...</th>
      <th>transmission_Semi-Auto</th>
      <th>fuelType_Electric</th>
      <th>fuelType_Hybrid</th>
      <th>fuelType_Other</th>
      <th>fuelType_Petrol</th>
      <th>car_type_coupe</th>
      <th>car_type_electric</th>
      <th>car_type_sedan</th>
      <th>car_type_sports</th>
      <th>car_type_suv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.750000</td>
      <td>0.313399</td>
      <td>0.215517</td>
      <td>0.111971</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.916667</td>
      <td>0.069281</td>
      <td>0.250000</td>
      <td>0.080163</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.833333</td>
      <td>0.293427</td>
      <td>0.275862</td>
      <td>0.098646</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.875000</td>
      <td>0.124650</td>
      <td>0.250000</td>
      <td>0.143778</td>
      <td>0.227273</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.750000</td>
      <td>0.184828</td>
      <td>0.275862</td>
      <td>0.096497</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10776</th>
      <td>0.833333</td>
      <td>0.190735</td>
      <td>0.258621</td>
      <td>0.104879</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10777</th>
      <td>0.833333</td>
      <td>0.200683</td>
      <td>0.215517</td>
      <td>0.117344</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10778</th>
      <td>0.875000</td>
      <td>0.119005</td>
      <td>0.344828</td>
      <td>0.080163</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10779</th>
      <td>0.750000</td>
      <td>0.210277</td>
      <td>0.051724</td>
      <td>0.126155</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10780</th>
      <td>0.875000</td>
      <td>0.277716</td>
      <td>0.215517</td>
      <td>0.111971</td>
      <td>0.303030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>10781 rows × 39 columns</p>
</div>



# Train Test Split


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```


```python
### Import our Machine Learning Algorithm
from sklearn.linear_model import LinearRegression
### Import our metric
from sklearn.metrics import mean_absolute_error


# Create a model object
linear_regressor = LinearRegression()

# Fit the object to our data (this is the training phase)
linear_regressor.fit(X_train, y_train)

# Create predictions with your newly trained model
linear_predictions = linear_regressor.predict(X_test)

# Measure the efficacy of your algorithm using your metric
mean_absolute_error(y_test, linear_predictions) 
```




    2766.8760709644484




```python
y_test.mean()
```




    22369.640578635015




```python
mean_absolute_error(y_test, linear_predictions) / y_test.mean()
```




    0.12368889259700756



Our mean predicted value is 12% above or below the mean value (12% Mean Error)


```python
### Import our Machine Learning Algorithm
from sklearn.ensemble import RandomForestRegressor
### Import our metric
from sklearn.metrics import mean_absolute_error


# Create a model object
random_forest_regressor = RandomForestRegressor(n_estimators=1000)    #Creates a forest of 1000 trees (Hyperparameter Tuning)

# Fit the object to our data (this is the training phase)
random_forest_regressor.fit(X_train, y_train)

# Create predictions with your newly trained model
random_forest_predictions = random_forest_regressor.predict(X_test)

# Measure the efficacy of your algorithm using your metric
mean_absolute_error(y_test, random_forest_predictions)
```




    1534.3394657206077




```python
mean_absolute_error(y_test, random_forest_predictions) / y_test.mean()
```




    0.06859026010395704



Our mean predicted value is 6.85% above or below the mean value (6.85% Mean Error)


```python
! pip install xgboost
```

    Collecting xgboost
      Downloading xgboost-1.6.1-py3-none-win_amd64.whl (125.4 MB)
         ------------------------------------ 125.4/125.4 MB 806.0 kB/s eta 0:00:00
    Requirement already satisfied: scipy in c:\users\autri ilesh banerjee\appdata\local\programs\python\python38\lib\site-packages (from xgboost) (1.8.1)
    Requirement already satisfied: numpy in c:\users\autri ilesh banerjee\appdata\local\programs\python\python38\lib\site-packages (from xgboost) (1.20.2)
    Installing collected packages: xgboost
    Successfully installed xgboost-1.6.1
    

    WARNING: There was an error checking the latest version of pip.
    


```python
### Import our Machine Learning Algorithm
from xgboost import XGBRegressor
### Import our metric
from sklearn.metrics import mean_absolute_error


# Create a model object
boost_model = XGBRegressor()

# Fit the object to our data (this is the training phase)
boost_model.fit(X_train, y_train)

# Create predictions with your newly trained model
boost_predictions = boost_model.predict(X_test)

# Measure the efficacy of your algorithm using your metric
mean_absolute_error(y_test, boost_predictions)

```




    1481.7090563675063




```python
mean_absolute_error(y_test, boost_predictions) / y_test.mean()
```




    0.06623749948770609



Our mean predicted value is 6.62% above or below the mean value (6.62% Mean Error)

# Hyperparameter Training


```python
from sklearn.model_selection import GridSearchCV
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

n_estimators = [1500, 1600]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [80, 90]
# Minimum number of samples required to split a node
min_samples_split = [5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
random_grid
```




    {'n_estimators': [1500, 1600],
     'max_features': ['auto'],
     'max_depth': [80, 90],
     'min_samples_split': [5],
     'min_samples_leaf': [1],
     'bootstrap': [True]}




```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 2, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
```

    Fitting 2 folds for each of 4 candidates, totalling 8 fits
    

    c:\users\autri ilesh banerjee\appdata\local\programs\python\python38\lib\site-packages\sklearn\ensemble\_forest.py:416: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features=1.0` or remove this parameter as it is also the default value for RandomForestRegressors and ExtraTreesRegressors.
      warn(
    




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=2, estimator=RandomForestRegressor(), n_jobs=-1,
             param_grid={&#x27;bootstrap&#x27;: [True], &#x27;max_depth&#x27;: [80, 90],
                         &#x27;max_features&#x27;: [&#x27;auto&#x27;], &#x27;min_samples_leaf&#x27;: [1],
                         &#x27;min_samples_split&#x27;: [5],
                         &#x27;n_estimators&#x27;: [1500, 1600]},
             verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=2, estimator=RandomForestRegressor(), n_jobs=-1,
             param_grid={&#x27;bootstrap&#x27;: [True], &#x27;max_depth&#x27;: [80, 90],
                         &#x27;max_features&#x27;: [&#x27;auto&#x27;], &#x27;min_samples_leaf&#x27;: [1],
                         &#x27;min_samples_split&#x27;: [5],
                         &#x27;n_estimators&#x27;: [1500, 1600]},
             verbose=2)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div></div></div></div>




```python
rf_random.best_params_
```




    {'bootstrap': True,
     'max_depth': 80,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 5,
     'n_estimators': 1600}




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

perfect_random_forest = RandomForestRegressor(n_estimators=1600, min_samples_split=5, min_samples_leaf=1, max_features='auto', max_depth=90, bootstrap=True)
perfect_random_forest.fit(X_train, y_train)

perfect_random_forest_predictions = perfect_random_forest.predict(X_test)

mean_absolute_error(y_test, perfect_random_forest_predictions)
```

    c:\users\autri ilesh banerjee\appdata\local\programs\python\python38\lib\site-packages\sklearn\ensemble\_forest.py:416: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features=1.0` or remove this parameter as it is also the default value for RandomForestRegressors and ExtraTreesRegressors.
      warn(
    




    1524.2450951532048




```python

```
