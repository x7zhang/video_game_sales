# video_game_sales

# Introduction
## Why I make this analysis?
7th GENERATION : Playstation 3 vs XBOX360 vs Nintendo Wii


8th GENERATION : Playstation 4 vs XBOXONE vs Nintendo WiiU


The aim is to run some visualisations on how some of the features in the dataset are correlated to one another as well as to provide some summary statistics and data analysis on the choice of genres and overall sales made by the different consoles to observe which one emerges with bragging rights.


```python
#Import related liberaries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib.patches as mpatches

```


```python
# import data
games = pd.read_csv('../Video Game Sales/Video_Games_Sales.csv')
games.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>82.53</td>
      <td>76.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>322.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>40.24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>Nintendo</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>35.52</td>
      <td>82.0</td>
      <td>73.0</td>
      <td>8.3</td>
      <td>709.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>32.77</td>
      <td>80.0</td>
      <td>73.0</td>
      <td>8</td>
      <td>192.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>Nintendo</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>31.37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




# Data Cleaning

## First we want to know whether there is a null item in it.


```python
games.isnull().any()
```




    Name                True
    Platform           False
    Year_of_Release     True
    Genre               True
    Publisher           True
    NA_Sales           False
    EU_Sales           False
    JP_Sales           False
    Other_Sales        False
    Global_Sales       False
    Critic_Score        True
    Critic_Count        True
    User_Score          True
    User_Count          True
    Developer           True
    Rating              True
    dtype: bool




```python
#use dropna metheod to drop nulls
games = games.dropna(axis=0)
```


```python
games.isnull().any()
```




    Name               False
    Platform           False
    Year_of_Release    False
    Genre              False
    Publisher          False
    NA_Sales           False
    EU_Sales           False
    JP_Sales           False
    Other_Sales        False
    Global_Sales       False
    Critic_Score       False
    Critic_Count       False
    User_Score         False
    User_Count         False
    Developer          False
    Rating             False
    dtype: bool




```python
games.Platform.unique()
```




    array(['Wii', 'DS', 'X360', 'PS3', 'PS2', '3DS', 'PS4', 'PS', 'XB', 'PC',
           'PSP', 'WiiU', 'GC', 'GBA', 'XOne', 'PSV', 'DC'], dtype=object)



# Correlation 


```python
corr = games.corr()
plt.subplots(figsize=(11,7))
sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            annot = True)
plt.xticks(rotation=75)

```




    (array([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5]),
     <a list of 9 Text xticklabel objects>)




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_10_1.png?raw=true)


# 7th Generation Console Comparation
### 7th Generation Console: 
            Playstation 3 vs XBOX360 vs Nintendo Wii


```python
#dataframe which only contain 7th generation console information:
console_7th = games[(games['Platform'] == 'Wii') | (games['Platform'] == 'PS3') |
                    (games['Platform'] == 'X360')]
```


```python
console_7th.shape
```




    (2106, 16)




```python
platWii = games[games['Platform'] == 'Wii']
platX360 = games[games['Platform'] == 'X360']
platPS3 = games[games['Platform'] == 'PS3']
```


```python
Wii_sales = platWii.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
PS3_sales = platPS3.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
X360_sales = platX360.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')

plt.figure(figsize=(15, 9))
ax = sns.pointplot(x = Wii_sales.index, y=Wii_sales.Global_Sales, color = 'blue')
ax = sns.pointplot(x = PS3_sales.index, y = PS3_sales.Global_Sales, color = 'red')
ax = sns.pointplot(x = X360_sales.index, y = X360_sales.Global_Sales, color = 'yellow')
plt.xticks(rotation=75)
plt.title('Pointplot of Global yearly Sales of the 7th Console', size=25)

Blue_patch = mpatches.Patch(color = 'blue', label = 'Wii Sales')
Red_patch = mpatches.Patch(color = 'red', label = 'PS3 Sales')
Yellow_patch = mpatches.Patch(color = 'yellow', label = 'X360 Sales')
plt.legend(handles=[Blue_patch, Red_patch, Yellow_patch], loc='upper right', fontsize = 16)

```




    <matplotlib.legend.Legend at 0x2087b3a47f0>




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_15_1.png?raw=true)


Attension year is 2009. From 2009, global sales for Wii become lowest, and the sales for X360 increased, and always bigger than Wii and PS3 in next 5 years. 

## PS3 and X360's Global Sale exceed Wii after 2009


# Why?


```python
console_7th_2009=test[(test['Year_of_Release']>2008.0) & (test['Year_of_Release']<2013.0)]
console_7th_2009.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>32.77</td>
      <td>80.0</td>
      <td>73.0</td>
      <td>8</td>
      <td>192.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>8</th>
      <td>New Super Mario Bros. Wii</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>14.44</td>
      <td>6.94</td>
      <td>4.70</td>
      <td>2.24</td>
      <td>28.32</td>
      <td>87.0</td>
      <td>80.0</td>
      <td>8.4</td>
      <td>594.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Kinect Adventures!</td>
      <td>X360</td>
      <td>2010.0</td>
      <td>Misc</td>
      <td>Microsoft Game Studios</td>
      <td>15.00</td>
      <td>4.89</td>
      <td>0.24</td>
      <td>1.69</td>
      <td>21.81</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>6.3</td>
      <td>106.0</td>
      <td>Good Science Studio</td>
      <td>E</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Wii Fit Plus</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>9.01</td>
      <td>8.49</td>
      <td>2.53</td>
      <td>1.77</td>
      <td>21.79</td>
      <td>80.0</td>
      <td>33.0</td>
      <td>7.4</td>
      <td>52.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Call of Duty: Modern Warfare 3</td>
      <td>X360</td>
      <td>2011.0</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>9.04</td>
      <td>4.24</td>
      <td>0.13</td>
      <td>1.32</td>
      <td>14.73</td>
      <td>88.0</td>
      <td>81.0</td>
      <td>3.4</td>
      <td>8713.0</td>
      <td>Infinity Ward, Sledgehammer Games</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>



## (1) Pulisher prefer 'X360 & PS3' than 'Wii'

### 2009-2015 Global Sales for different genre in different platform 


```python
global_sales_by_genre = console_7th_2009.groupby('Genre').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
plt.subplots(figsize=(11,7))
ax = sns.barplot(x=global_sales_by_genre.index, y=global_sales_by_genre.Global_Sales)

```


![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_22_0.png?raw=true)


The highest global sale of top 8 are 'Action, Shooter, Sports, Misc, Role-Playing, Racing, Platform, Fighting'. There percentage is higher to XXX%.


```python
#each genre percentage for total global sales
```

### Rating arrangement for different platform


```python
ratingPlatform = console_7th_2009.groupby(['Platform','Rating']).Platform.count()
ratingPlatform.unstack().plot(kind='bar', stacked=True, grid=False)
plt.title('Stacked Barplot of Rating Types of the 7th Gen Consoles(2009-2016)')
plt.ylabel('Totle Number')
```




    <matplotlib.text.Text at 0x20804f49f98>




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_26_1.png?raw=true)

Wii  => [E, E10+, T]
PS3  => [T, M]
X360 => [T, M]
So we can divide platform in general: Wii is focus on children market.
X360 and PS3 focus on adult and teenager market.

## (2) Different Rating Market
### Split game genre in different rating


```python
genreRating = console_7th_2009.groupby(['Genre', 'Rating']).Genre.count()
ax =genreRating.unstack().plot(kind='bar', stacked=True, grid=False)
plt.title('Stacked Barplot of Rating Types of the Genre(2009-2016)')
plt.ylabel('Genre')
```




    <matplotlib.text.Text at 0x208038724e0>




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_29_1.png?raw=true)


The highest global sale of the top 8's rating as follow:

    Action  => [M, T]
    Shooter => [M, T]
    sports  => [E, E10+]
    Misc    => [T, E10+]
    Role-Playting =>[T, M]
    Racing  => [E, E10+]
    Platform => [E10+, E]
    Fighting => [T, M]

This plots means adult and teenager market are the main market for publisher, they prefer to publish '[M, T]' games during 2009 to 2012.


```python
genreRating = console_7th_2009.groupby(['Genre', 'Rating']).Genre.count()
ax = sns.heatmap(genreRating.unstack())
plt.title('Stacked Barplot of Rating Types of the Genre(2009-2016)')
plt.ylabel('Genre')
```




    <matplotlib.text.Text at 0x20803758080>




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_33_1.png?raw=true)


    T => ['Action', 'Fighting', 'Misc', 'Role-Playing', 'shooter']
    M => ['Action', 'Role-Playing', 'Shooter']
    E10+ => ['Action', 'Misc', 'Platform', 'Racing', 'Sports']
    E => ['Sports', 'Racing']


E and E10+ genre of games are mainly distributed in 'Action', 'Misc', 'Platform', 'Racing', 'Sports', 'Sports'. 



```python
genre_Sales = console_7th_2009.groupby(['Genre', 'Platform']).Global_Sales.sum()
ax = genre_Sales.unstack().plot(kind='bar', stacked=True, grid=False)
plt.title('Stacked Barplot of Sales per Game Genre (2009-2012)')
plt.ylabel('Sales')
```




    <matplotlib.text.Text at 0x20803b4a438>




![png](https://github.com/x7zhang/video_game_sales/blob/master/graphs/output_36_1.png?raw=true)



```python

```


In 'Satcked Barplot of Sales per Game Genre', we can find, except 'Misc', 'Platform', Publisher public Wii games significantly less than 'PS3' and 'X360'.

Wii is focus on chidren market， this theory has been proved before.
So we are sure that Publisher are willing to release 'X360 and PS3' games, that is one of reasons why after 2009, Wii games' global sale are lower than X360 and PS3. 



```python
genre_Sales.unstack().sum()
```




    Platform
    PS3     784.30
    Wii     658.98
    X360    853.30
    dtype: float64



 It seems that for both the PS3 and XB360, their 2 main genres were Action and Shooter games which as we know was the case as they appeal more to the hardcore, action-oriented gamer. The Wii on the other hand, focused on the genre of Sports, Platformers as well as some other Misc games.


```python

```

# 8th Generation Console Comparation
### 8th Generation Console: 


```python
#dataframe which only contain 7th generation console information:
console_8th = games[(games['Platform'] == 'WiiU') | (games['Platform'] == 'PS4') |
                    (games['Platform'] == 'XOne')]
console_8th.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>2015.0</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>6.03</td>
      <td>5.86</td>
      <td>0.36</td>
      <td>2.38</td>
      <td>14.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>2014.0</td>
      <td>Action</td>
      <td>Take-Two Interactive</td>
      <td>3.96</td>
      <td>6.31</td>
      <td>0.38</td>
      <td>1.97</td>
      <td>12.61</td>
      <td>97.0</td>
      <td>66.0</td>
      <td>8.3</td>
      <td>2899.0</td>
      <td>Rockstar North</td>
      <td>M</td>
    </tr>
    <tr>
      <th>77</th>
      <td>FIFA 16</td>
      <td>PS4</td>
      <td>2015.0</td>
      <td>Sports</td>
      <td>Electronic Arts</td>
      <td>1.12</td>
      <td>6.12</td>
      <td>0.06</td>
      <td>1.28</td>
      <td>8.57</td>
      <td>82.0</td>
      <td>42.0</td>
      <td>4.3</td>
      <td>896.0</td>
      <td>EA Sports</td>
      <td>E</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Star Wars Battlefront (2015)</td>
      <td>PS4</td>
      <td>2015.0</td>
      <td>Shooter</td>
      <td>Electronic Arts</td>
      <td>2.99</td>
      <td>3.49</td>
      <td>0.22</td>
      <td>1.28</td>
      <td>7.98</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Call of Duty: Advanced Warfare</td>
      <td>PS4</td>
      <td>2014.0</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>2.81</td>
      <td>3.48</td>
      <td>0.14</td>
      <td>1.23</td>
      <td>7.66</td>
      <td>83.0</td>
      <td>39.0</td>
      <td>5.7</td>
      <td>1443.0</td>
      <td>Sledgehammer Games</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
platWiiU = games[games['Platform'] == 'WiiU']
platXOne = games[games['Platform'] == 'XOne']
platPS4 = games[games['Platform'] == 'PS4']
```


```python
WiiU_sales = platWiiU.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
PS4_sales = platPS4.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
XOne_sales = platXOne.groupby('Year_of_Release').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')

plt.figure(figsize=(15, 9))
ax = sns.pointplot(x = WiiU_sales.index, y=WiiU_sales.Global_Sales, color = 'blue')
ax = sns.pointplot(x = PS4_sales.index, y = PS4_sales.Global_Sales, color = 'red')
ax = sns.pointplot(x = XOne_sales.index, y = XOne_sales.Global_Sales, color = 'yellow')
plt.xticks(rotation=75)
plt.title('Pointplot of Global yearly Sales of the 7th Console', size=25)

Blue_patch = mpatches.Patch(color = 'blue', label = 'WiiU Sales')
Red_patch = mpatches.Patch(color = 'red', label = 'PS4 Sales')
Yellow_patch = mpatches.Patch(color = 'yellow', label = 'XOne Sales')
plt.legend(handles=[Blue_patch, Red_patch, Yellow_patch], loc='upper right', fontsize = 16)

```




    <matplotlib.legend.Legend at 0x2087cd9ad30>




![png](output_63_1.png)


It is obvious just by one look that the PS4 global sales exceed those of BOTH the WiiU and XOne combined. This is a very marked deviation from its predecessor's performance in the 7th Gen when the PS3 and XB360 where neck to neck in sales performance over the years. So how can be explain this dominance this time round?

## Question: Whether we can invest in PS4 in further?


```python
usercountYear = console_8th.groupby(['Year_of_Release', 'Platform']).User_Count.sum()

ax = usercountYear.unstack().plot(kind='bar', stacked=True, grid=False)


```


![png](output_66_0.png)


The first impression is after PS4 released in 2013, the total count of user increased sharply util 2016. In 2016,  the sum of user count is lower than 2014, although the total number of user count is decreased, the total user for PS4 is almost maintain the same user count level as in 2014. 

It showns market share for PS4 is increased.



```python
globalSales_by_genre = console_8th.groupby('Genre').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')
plt.subplots(figsize=(11,7))
ax = sns.barplot(x=globalSales_by_genre.index, y=globalSales_by_genre.Global_Sales)

```


![png](output_68_0.png)



```python
 releaseAction = console_8th.groupby('Year_of_Release').app({})
```
