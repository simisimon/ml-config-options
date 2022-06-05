# coop
current:
1) class_scraper.py
   - extract classes incl. parameter from doc
   - save into json file
2) obj_selector.py
   - extract keyword from import lines from project source code
   - loop through source code and extract lines containing those keywords
   - loop through those preselected lines and extract lines that contain classes from step 1)


_______________
old:
1) extract.py
   - extract information from ML-lib docs
2) analyze_import.py 
   - extract keywords from import lines of the source code that use those ML-libs
   - loop through source code and extract lines containing those keywords


current status:
   - identified lines of code that use ML-libs

next steps: 
   - distinguish potential config options from irrelevant code in a most generic way

Requirements: 
   - Python 3.9
   - BeautifulSoup
   - GitPython
   - Scalpel
     - astor
     - networkx
     - graphviz