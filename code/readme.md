# All the code necessary to do the scraping, and subsequent data analysis is here, in two jupyter notebooks.

Requires (python2): pandas, numpy, scikit learn, urllib2, BeautifulSoup


## ```online_social_function.ipynb```

* ```G2GAnalysis``` is the class object that collects, contains and manipulates all the user and question data.
    * ```G2GAnalysis.runScrape()``` begins a new scrape of the Wikitree G2G website. 
    
## ```analyse_and_graph.ipynb```

* ```G2GDat``` is the class object that loads saved data, and processes it into a pandas dataframe for subsequent analysis.
* Several predictive pipelines are implemented
