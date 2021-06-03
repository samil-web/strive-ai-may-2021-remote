import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

response = requests.get("https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996")

print("Response: ", response)

page = response.content

soup = BeautifulSoup(page, "html.parser")

periods = [period.get_text(separator = ' ') for period in soup.find_all('p', class_ = 'period-name')]
description = [period.get_text(separator = ' ') for period in soup.find_all('p', class_ = 'short-desc')]
temperatures = [period.get_text(separator = ' ') for period in soup.find_all('p', class_ = 'temp')]

def celsius_scale(temperature_string):
    temperature = int(temperature_string.split()[1])
    return str(round((temperature-32)*(5/9), 1)) + u" \N{DEGREE SIGN}C"

temperatures_celsius = []
for temperature in temperatures:
    temperatures_celsius.append(celsius_scale(temperature))

# print(temperatures_celsius)

# print(np.array([periods, description, temperatures]).transpose())
df = pd.DataFrame(data = np.array([periods, description, temperatures_celsius]).transpose(), columns = ['Day','Description','Temperatures'])
print(df)