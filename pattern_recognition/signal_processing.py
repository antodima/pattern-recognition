#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def autoregressive_analysis_appliances():
    """
    +----------------+
    | Assignment 1.1 |
    +----------------+
    
    Perform an autoregressive analysis of the “Appliances” column of the dataset 
    which measures the energy consumption of appliances across a period of 4.5 months. 
    Fit an autoregressive model on the first 3 months of data and estimate performance 
    on the remaining 1.5 months. Try out different configurations of the autoregressive 
    model (e.g. experiment with AR models of order 3, 5 anarmad 7). 
    You can use the autoregressive model of your choice (AR, ARMA, ...) 
    and perform data pre-processing operations, if you wish (not compulsory).
    """
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.arima_model import ARIMA
    
    # https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#
    appliance_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 1])
    #print(energy_data.describe())
    #plt.plot(energy_data)
    #plot_pacf(energy_data["Appliances"])
    train = appliance_series.loc['2016-01-01':'2016-03-31']
    test = appliance_series.loc['2016-04-01':]
    p, q = 1, 8
    arma_model = ARIMA(train, order=(p, 0, q))
    arma_model_fit = arma_model.fit()
    pred = arma_model_fit.predict(start='2016-04-01 00:00:00', end='2016-05-27 18:00:00')
    print('MSE:', mean_squared_error(test, pred))
    plt.plot(test)
    plt.plot(pred)
    plt.show()


def correlation_temperatures():
    """
    +----------------+
    | Assignment 1.2 |
    +----------------+
    
    Perform a correlation analysis on the temperature data in the dataset 
    (i.e. the columns marked as Ti). It is  sufficient to pick up just one 
    of the 10 sensors and show the cross-correlation plot against the remaining 9 
    sensors (make a plot for each of the sensors, but try to put 4/5 of them 
    in the same slide in a readable form).
    """
    from statsmodels.graphics.tsaplots import plot_pacf
    
    temperature_t1_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 3])
    temperature_t2_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 5])
    temperature_t3_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 7])
    temperature_t4_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 9])
    temperature_t5_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 11])
    temperature_t6_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 13])
    temperature_t7_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 15])
    temperature_t8_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 17])
    temperature_t9_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 19])
    temperature_t10_series = pd.read_csv('datasets/energy_data/energydata_complete.csv', 
                              header=0, index_col=0, parse_dates=True, usecols=[0, 21])
    # the autocorrelation is the autocovariance with lag −N ≤ τ ≤ N 
    # w.r.t. autocovariance with τ = 0
    pd.plotting.autocorrelation_plot(temperature_t1_series)
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t2_series.values.flatten(), mode='full'), label='t1 vs t2')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t3_series.values.flatten(), mode='full'), label='t1 vs t3')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t4_series.values.flatten(), mode='full'), label='t1 vs t4')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t5_series.values.flatten(), mode='full'), label='t1 vs t5')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t6_series.values.flatten(), mode='full'), label='t1 vs t6')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t7_series.values.flatten(), mode='full'), label='t1 vs t7')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t8_series.values.flatten(), mode='full'), label='t1 vs t8')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t9_series.values.flatten(), mode='full'), label='t1 vs t9')
    plt.plot(np.correlate(temperature_t1_series.values.flatten(), 
                          temperature_t10_series.values.flatten(), mode='full'), label='t1 vs t10')
    plt.legend(loc='upper right')
    plt.show()


class SimplePitchDetector(object):
    """
    +----------------+
    | Assignment 1.3 |
    +----------------+
    
    The autocorrelation is useful for finding repeated patterns in a signal. 
    For example, at short lags, the autocorrelation can tell us something about 
    the signal's fundamental frequency (i.e. the pitch). Implement a very simple 
    pitch detector from music signals using the autocorrelation function. 
    Following up this reference article (Section 2.A) [http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf]:
    
    The "autocorrelation method" chooses the highest non-zero-lag peak by exhaustive 
    search within a range of lags. Obviously if the lower limit is too close to zero, 
    the algorithm may erroneously choose the zero-lag peak.
    
    So basically you need to find the lag corresponding to the maximum in the autocorrelogram, 
    avioding to pick up the trivial lag 0. Given a signal sampled at frequency Fs 
    and the maximum lag being τ the corresponding pitch is Fs÷τ.
    
    To complete the assignment apply the analysis above to (at least) one of the WAVs 
    in this repository (https://github.com/stevetjoa/musicinformationretrieval.com/tree/gh-pages/audio).
    In Python you can import WAVs (and acces several other 
    music-related functions), using the LibROSA library (https://librosa.github.io/librosa/).
    
    >> "~/anaconda3/bin/conda install -c conda-forge librosa"
    """
    
    def __init__(self, file_path):
        import librosa
        self.timeseries, self.sr = librosa.load(file_path)

    def autocorrelation(self, lag=1, window_size=2):
        tau = lag
        w = window_size
        rates = [0] * len(self.timeseries)
        for t in range(0, len(self.timeseries)):
            for j in range(t+1, t+w):
                if j+tau >= len(self.timeseries)-1: continue
                rates[t] += self.timeseries[j] * self.timeseries[j+tau]
        return rates
    
    def detect_pitch(self, window_size=2, lag_interval=(1, 10)):
        w = window_size
        max_rates = []
        pitches = []
        for tau in range(lag_interval[0], lag_interval[1]+1):
            tau_max = w
            rates = self.autocorrelation(tau, w)
            rates = [(r * (1 - tau/tau_max)) if (r <= tau_max) else 0 for r in rates]
            pitch = self.sr / tau
            pitches.append(pitch)
            max_rates.append(max(rates))
        return max_rates, pitches
