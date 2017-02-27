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

1. Using findNaN function that I wrote, I discovered that there four households that containted null data.  It appears that these null values comprise an entire months worth of data.  In all likely hood these occupants were either moving in or moving out their houses... thus this dataset only captured their first month or last month of electricity usage.  
2. 

### Which houses own and EV?
Approximately 1/3 of the training set households owned an EV, or 485/1586 = 30.5% if you prefer decimals.  
