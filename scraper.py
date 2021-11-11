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


title = soup.title
titleText = title.get_text()

body = soup.find('torch', class_='pytorch-body')

section = soup.find('section', class_="css-1r7ky0e")
for elem in section:
    div1 = elem.findAll('div')
    for x in div1:
        div2 = elem.findAll('div')
        for i in div2:
            text = i.find('p').get_text()
            print (text)
            print("----------")