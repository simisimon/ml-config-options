# Find example

# Import the libraries BeautifulSoup
# and os
from bs4 import BeautifulSoup as bs
import os

# Open the HTML in which you want to
# make changes
#html = 'https://pytorch.org/docs/stable/torch.html'

# Parse HTML file in Beautiful Soup
#soup = bs(html, 'html.parser')

# Obtain the text from the widget after
# finding it
#find_example = soup.find("p", {"id": "torch"}).get_text()

# Printing the text obtained received
# in previous step
#print(find_example)

from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://pytorch.org/docs/stable/torch.html"
html = urlopen(url)
soup = BeautifulSoup(html, "html.parser")

tables = soup.find_all('table')
torch_dict = {}
for table in tables:
    for row in table.find_all('tr'):
        row_cut = row.get_text()[:-1]
        k = row_cut.split("\n")
        torch_dict[k[0]] = k[1]

print(torch_dict)
