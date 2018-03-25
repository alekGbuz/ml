import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
# Modify the cities table by adding a new boolean column that is True if and only if both of the following are True:
# The city is named after a saint.
# The city has an area greater than 50 square miles.
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & (cities['City name'].str.startswith('San'))
print(cities.head())

