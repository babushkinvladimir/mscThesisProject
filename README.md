<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/babushkinvladimir/mscThesisProject/blob/master/IMG/Logo.png" align="center" height="100"></a>
</p>

<div align="center"> 

#### Universitatea de Stat din Moldova

#### State University of Moldova

# DEVELOPMENT OF AUTOMATIC LEARNING ALGORITHMS FOR PRICE AND LOAD PREDICTION IN ELECTRICAL NETWORK


##### VLADIMIR BABUSHKIN

###### TEZELOR DE LICENŢĂ/MASTER
###### MSC THESIS


## ABSTRACT
</div>
Nowadays, the electricity market is becoming more diverse and deregulated due to the gradual introduction of smart grids. A smart grid analyzes information about electricity supply and demand to optimize the electric power production. This requires the knowledge of future electricity load that allow the Independent Power Producers to schedule the electric power generation to meet the peak demand and minimize the production costs. The IPPs can also implement the dynamic pricing to incentivize the end-users shifting the electricity consumption to off-peak hours. In other word, consumers, knowing the future electricity prices, can adjust their consumption profile to achieve an optimal utilization of electric power at the lowest cost. Other entities of the electricity market e.g. retailers and traders are also interested in electricity price prediction to optimize their financial operations. Thus, the accurate forecasting of price and load is essential for maintaining a stable interplay between demand and supply in the dynamic electricity market. 

This work aims to investigate non-linear methods for time series prediction and to compare them with a state-of-the-art deep learning methods. Deep learning networks are capable of inferring hidden temporal dependences and trends as well as intrinsic factors, affecting the dynamicity of electricity load and price time series. We explore the performance of Long-Short Term Memory and Convolutional Neural Networks for predicting power balance in Moldova as well as electricity load and price for New York City and New South Wales. We also propose a novel deep learning approach for day-ahead electricity price forecasting from historical price/load data and the predicted day-ahead load value. The proposed system can be implemented in smart grid settings for online and offline electricity price prediction for forecasting horizons of different lengths, ranging from a single day to weeks and months. 
<div align="center">
  
## DATASETS
</div>

  - Monthly electricity balance data provided by “Moldelectrica” Transmission System Operator [1] for Moldova
  - Electrical load and price data provided by New York Independent System Operator [2] for New York City district (N.Y.C.)
  - Electrical load and price data provided by Australian Energy Market Operator [3] for New South Wales (N.S.W.)
  - COVID-19 New York City data [4]
  - COVID-19 New South Wales, Australia data [5]
  
<div align="center">
  
## APIs
</div>

  - Keras  [6] -- a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. We use Keras to develop sequential CNN and LSTM models.
  - Scikit-learn [7] is a free software machine learning library for Python.
  - k-medoids clustering package [8] within the scikit-learn-extra package. Scikit-learn-extra is a Python module for machine learning that extends scikit-learn. It includes algorithms that are useful but do not satisfy the scikit-learn inclusion criteria, for instance due to their novelty or lower citation number.
  - MiniSom [9] a minimalistic Numpy based implementation of the Self Organizing Maps (SOM)
  
## REFERENCES

[1] "Pronosticul lunar: Pronosticul transportului de energie şi putere electrică de către sistemul energetic al Republicii Moldova, împărţit pe Furnizori," Moldelectrica Operatorul sistemului de transport, http://www.moldelectrica.md/ro/electricity/monthly_forecast

[2] "Energy market & operational data," New York Independent System Operator, https://www.nyiso.com/energy-market-operational-data.

[3] "Aggregated price and demand data," Australian Energy Market Operator, https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data

[4] "COVID-19: Data," NYC government,  https://www1.nyc.gov/site/doh/covid/covid-19-data.page

[5] "COVID-19 cases by notification date," The government of New South Wales , https://data.nsw.gov.au/data/dataset/covid-19-cases-by-location/resource/21304414-1ff1-4243-a5d2-f52778048b29

[6] F. Chollet, "Keras," 2015, https://keras.io

[7] F. Pedregosa, V. Gael, G. Alexandre, M. Vincent, T. Bertrand, G. Olivier, M. Blondel, P. Prettenhofer, W. Ron, D. Vincent, V. Jake, P. Alexandre, C. David and B. Matthieu, "Scikit-learn: Machine learning in Python.," Journal of Machine Learning Research, p. 2825–2830, 2011. 

[8] T. Erkkilä, A. Lehmussola, K. Kiełczewski and Z. Dufour, "K-medoids clustering," 2019. https://github.com/scikit-learn-contrib/scikit-learn-extra/blob/master/sklearn_extra/cluster/_k_medoids.py.

[9] V. Giuseppe, "MiniSom," https://github.com/JustGlowing/minisom


