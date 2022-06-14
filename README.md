# Extract ML Configurations from Python Source Code
This tool can thus be used to identify ML algorithms and the configuration options.

In machine learning, configuration options exist in form of hyperparameters that can be used to control ML algorithms. This tool presents an approach to extract ML configuration options from Python source code. Using static code analysis, this approach can be used to locate the configuration options of popular ML libraries in arbitrary software systems. In addition, techniques from data flow analysis can be used to determine possible configuration values even in the presence of variable hyperparameters. 

Currently this tool can be used for the following librarys:
   - TensorFlow
   - PyTorch
   - scikit-learn
   - Mlflow

### Step 1：Install requirements

Requirements: 
   - Python >= 3.9
   - BeautifulSoup
   - GitPython
   - Scalpel
     - astor
     - networkx
     - graphviz

### Step 2：Scrape classes
Run the following command to scrape classes and parameters from the API of the library, e.g. scikit-learn:
```
python3 scraping.py scikit-learn
```

### Step 3：Extract ML classes and parameters
Run the following command to extract the ML algorithms, hyperparameters, and potential values of variable hyperparameters
```
python3 main.py https://github.com/user/project scikit-learn
```

Results will be stored in the 'output' directory.
