## jpx-tokyo-stock-exchange-prediction

competencia https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction

You can calculate the Target column from the Close column; it's the return from buying a stock the next day and selling the day after that.

```
pip install kaggle --upgrade
descargar kaggle.json desde la cuenta (create api token)
mv /home/{user}/Downloads/kaggle.json /home/{user}/.kaggle
chmod 666 /home/{user}/.kaggle/kaggle.json 
kaggle competitions download -c jpx-tokyo-stock-exchange-prediction
```

una buena notebook para entender la competencia aca, tiene funciones piolas para reutilizar
  - https://www.kaggle.com/code/chumajin/english-ver-easy-to-understand-the-competition

notebook en wip de un flaco q figura como kaggle expert, para chusmear
  - https://www.kaggle.com/code/cv13j0/jpx-tokyo-stock-exchange-prediction-xgboost

recursos Mutt
  - https://gitlab.com/mutt_data/handbook/-/blob/master/exercises/eda/readme.md
  - https://gitlab.com/mutt_data/handbook/-/blob/master/exercises/eda/notebooks/exploratory_data_analysis.ipynb

MEAN model 
  - https://www.kaggle.com/code/paulorzp/mean-model-jpx/notebook?scriptVersionId=92406307

https://www.kaggle.com/code/cv13j0/jpx-tokyo-stock-exchange-prediction-xgboost

7506-zonaprop
  - https://github.com/nbuzzano/7506-zonaprop

https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/discussion/317395

What does your machine learning tool stack look like? ([fuente](https://analyticsindiamag.com/how-i-became-an-ml-hackathon-grandmaster/?utm_source=onesignal&utm_medium=push&utm_campaign=2022-04-21-How-I-became-an)
)

  - I use Google Colab or Kaggle Kernel platform to write the code from scratch. 

  - From a tool’s point of view, I use Visual Studio, H2o.ai, Weka, Jupyter (Anaconda), Google Colab, Kaggle Kernel, System GPU, MS SQL, Google Cloud Platform, Amazon Machine learning, etc. Sometimes, I use Spyder and PyCharm for ML projects. 

  - From a framework point of view, I use scikit-learn, PyTorch, TensorFlow, fast.ai, Keras, Google Cloud ML Engine, and Spark MLlib. 

  - From a project point of view, I prefer to use local systems.


MISC
  - Pandas Profiling: https://pypi.org/project/pandas-profiling/ & https://github.com/ydataai/pandas-profiling/tree/develop/examples
  - https://github.com/orga-de-datos/practicas/tree/master/clases
  - notebook time series EDA https://github.com/PacktPublishing/Hands-on-Exploratory-Data-Analysis-with-Python/blob/master/Chapter%208/Time_Series_Analysis.ipynb
  - https://pypi.org/project/sweetviz/
  - EDA step by step https://jovian.ai/evilmhany/egy-stock-market-eda/v/1

Hacer research de: 
- EDA in stocks market
- EDA in time series
- EDA stock market paper

https://machinelearningmastery.com/time-series-data-visualization-with-python/

Live Time series > https://www.youtube.com/playlist?list=PLZoTAELRMXVNty3jyJkYXuyQY3lMSpr3b

https://machinelearningmastery.com/?s=time+series&post_type=post&submit=Search

https://machinelearningmastery.com/time-series-forecasting/
https://machinelearningmastery.com/how-to-develop-a-skilful-time-series-forecasting-model/
https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/

- cuando hay split de acciones, da lugar a nuevo inversores. tendiamos que tener una feature some how aca ?

- las rows que tiene 1 col vacia ( precio de apertura por ej) deberiamos limpiarla ? 

Approximately 20% of trading dates have all 2000 stocks' records. ( no todos los stocks estan en todo el periodo 2017-01-04 ~ 2021-12-03 )

Over 90% of stocks have only one date with missing prices.

el 2020-10-01 estuvo la caida de la bolsa de china (https://www.jpx.co.jp/english/corporate/news/news-releases/0060/20201019-01.html) despues o antes de eso las acciones se comportaron de forma extrania ?

OHLCV - open high low close values

=============

Target mean is a right-skewed distribution. Also, it has a large kurtosis, exhibiting tail data exceeding the tails of the normal distribution. For investors, high kurtosis of the return distribution implies the investor will experience occasional extreme returns (either positive or negative). This phenomenon is known as kurtosis risk.

entender la mean y kurtosis del periodo (por ejemplo los ultimos 10 dias) para entender el sentimiento del inversor y saber si estan por tomar un riesgo y comprar mucho o vender muchas acciones ( cosa que te mueve el precio)

tomar una ventana de tiempo y hacerlo por la compania y/o mezclando todas

=============

hay que saber de que rubro son las stock, no es lo mismo acciones de IT , que de campo 

=============

The joint plot of number of dates (i.e., records) per stock and target mean distribution shows that target mean increases proportionally to the number of dates. Moreover, the dispersion of target mean seems to be larger when the number of dates is smaller.

Por esto que comenta arriba, pareciera que las acciones que tiene pocas fechas registradas, tienen una dispersion de target mas elevada , 
- me queda la duda, si tuviera igual cantidad de muestras la dispersion disminuiria ?
- puedo agrupar por sector al que corresponde el stock ( IT, metalera, cerealeras, etcs) y saber si la dispersion se corresponde con ciertos sectores ?

=============

A slight increase of dispersion is observed when number of stocks per date becomes larger.

parece que te aumenta la el desvio standard del target en el mercado a medida que hay mas stock

=============
https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/discussion/317395

Section-specific series show that the business activity in prime market is more vigorous than the others. Following are just two examples for illustration

=============

Como incluir la info de opciones te la debo

========================================

https://www.kaggle.com/code/junjitakeshima/jpx-add-new-features-eng

https://www.kaggle.com/code/lucasmorin/jpx-macro-data-from-public-apis/notebook

https://www.kaggle.com/code/bowaka/jpx-buying-strategy-backtest/notebook

https://www.kaggle.com/code/satoshidatamoto/jpx-volatility-features/notebook