import pprint
from urllib.request import urlopen
from bs4 import BeautifulSoup
import torch

def extract_doc(url_suffix):
    # Open the HTML
    torch_dict = {}
    for u in url_suffix:
        link = url + u
        try:
            html = urlopen(link)
        except:
            continue

        # Parse HTML file in Beautiful Soup
        soup = BeautifulSoup(html, "html.parser")
        # Obtain the text from the widget after finding it
        tables = soup.find_all("table")
        u = u.split(".html")[0]
        torch_dict[u] = {}

        for table in tables:
            for row in table.find_all("tr"):
                row_cut = row.get_text()[:-1]
                k = row_cut.split("\n")
                torch_dict[u][k[0]] = k[1]

    return torch_dict

def get_torch_url(url):
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")

    modules = soup.find_all('div', {'class': 'toctree-wrapper compound'})[2]
    modules = modules.find_all('a')
    url_suffix = []
    for module in modules:
        url_suffix.append(module['href'])

    return url_suffix

url = "https://pytorch.org/docs/stable/"
url_suffix = get_torch_url(url)
torch_dict = extract_doc(url_suffix)
pprint.pprint(torch_dict.keys())

