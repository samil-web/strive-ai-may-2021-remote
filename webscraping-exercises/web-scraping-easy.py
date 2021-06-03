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
# print(periods)
# print(description)
# print(temperatures)
print(np.array([periods, description, temperatures]).transpose())
df = pd.DataFrame(data = np.array([periods, description, temperatures]).transpose(), columns = ['Day','Description','Temperatures'])
print(df)