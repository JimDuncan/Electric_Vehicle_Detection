# Electric Vehicle Detection
Detecting the presence of an Electic Vehicle that is connected to the grid.

<img src = "http://st.automobilemag.com/uploads/sites/11/2015/01/2014-BMW-i3-eDrive-rear-side-view-charging1.jpg" width = "250">

##Project Summary
This project utilizes an open data set from GridCure.  The intention of this project is the following:

* Interesting aspects about this dataset
* Identify which households own an Electric Vehicle(EV)
* Determine the probability that an EV is charging at any given interval


###Interesting aspects of the dataset
I am going to tackle this one first because it affected how I treated cleaning and ultimately how I trained my algorithms on it. 

1. Using findNaN function that I wrote, I discovered that there were four households that containted null data in the training set and two households in the test set.  Given that these houses that had null data only comprimised less than 0.5% of the data I felt comfortable eliminating them from the datasets.  It appears that these null values comprise an entire months worth of data.  In all likely hood these occupants were either moving in or moving out their houses... thus this dataset only captured their first month or last month of electricity usage. 
2. Using the average usage of each household as a representation for the training set, it appears over the course of two weeks that each day there is spike in electricity.  Graph below.
![alt tag] (https://github.com/ajduncan3/Electric_Vehicle_Detection/blob/master/Graphs%20and%20Pictures/Average%20household%20use%20over%20two%20weeks.png)

Filtering down into a single days worth of electicity usage, it becomes apparent that usage begins to ramp up in the early morning, remains steady throughout mid morning and early afternoon, peaks in the evening and then finally falls off precipitously right after peak usage. This makes intuitive sense, there is likely minimal electricity use when everyone in a house is asleep and usage will start to increase as people wake up and start to make breakfast,use the toaster, stove, etc... and electricity use will be at it's peak in the evening when everyone in the household is home and using all their computers, TV's,running the air conditioner, and charging their EV if they own one. Graph below.
![alt tag](https://github.com/ajduncan3/Electric_Vehicle_Detection/blob/master/Graphs%20and%20Pictures/Average%20household%20use%20in%20one%20day.png)

### Which houses own an EV?
Approximately 1/3 of the training set households owned an EV, or 485/1586 = 30.5% if you prefer decimals.  The House ID can be found in the the pickled file 'Houses_that_own_ev.pkl' located in this repo's data_files folder.  Or just go to the link below.

https://raw.githubusercontent.com/ajduncan3/Electric_Vehicle_Detection/master/data_files/Houses_that_own_ev.pkl

Now, over any one interval it appears that at most 16% of households have an EV plugged in and given that almost two sigmas worth of the households has their EV plugged in less than 10 % of the time... this problem appears to be a minority class detection problem.  Very similar to a fraud detection problem, I needed to detect with high recall an occurrence that only happens a small percentage of the time.  I used a variety a techniques ranging from using the synthetic minority oversampling technique(SMOTE) to a gradient boosted tree classifier.  Yet no matter what technique or combination of techniques I couldn't crack a 5% recall.  That is not good.  Enter Neural Networks and the Long Short Term Memory(LSTM) layer strucutre.

### Determine the probability an EV is charging at a given interval
