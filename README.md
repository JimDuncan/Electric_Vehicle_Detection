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
2. Using the average usage of each household as a representation for the training set, it appears over the course of two weeks that each day there is spike in electricity.  Graph below.
![alt tag] (https://github.com/ajduncan3/Electric_Vehicle_Detection/blob/master/Graphs%20and%20Pictures/Average%20household%20use%20over%20two%20weeks.png)

Filtering down into a single days worth of electicity usage, it becomes apparent that usage begins to ramp up in the early afternoon and peaks in the evening.. then falls off precipitously right before early morning. This makes intuitive sense, most people are arriving home from work in the afternoon, and begin to use their appliances for their evening.  I would assume a typical person comes home, turns on their heater/air conditioner, turns on the TV, starts cooking dinner, users their computer,  maybe plugs in their Electric Vehicle if they own one, etc... Absent of that individual being home those appliances are probably not going to be pulling from the power grid.  Graph below.
![alt tag](https://github.com/ajduncan3/Electric_Vehicle_Detection/blob/master/Graphs%20and%20Pictures/Average%20household%20use%20in%20one%20day.png)

### Which houses own and EV?
Approximately 1/3 of the training set households owned an EV, or 485/1586 = 30.5% if you prefer decimals.  
