from utils.outliers import *
from utils.load_data import *
from utils.diagnostics import *


#Plots

#subject 1 fMRI data 
data = get_image(3,9).get_data()
std = vol_std(data)


# RMS outliers plot 

rms = vol_rms_diff(data)



outliers_rms = iqr_outliers(rms)[0]
ranges = iqr_outliers(rms)[1]

outliers_values = []
for i in outliers_rms:
	outliers_values.append(rms[i])

plt.figure()
plt.plot(rms)
plt.plot(outliers_rms,outliers_values,"ro", marker = "o")
plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")

plt.savefig('outliers RMS sub9 run3')



### subject 1 run 1 

data = get_image(1,1).get_data()
std = vol_std(data)

rms = vol_rms_diff(data)



outliers_rms = iqr_outliers(rms)[0]
ranges = iqr_outliers(rms)[1]

outliers_values = []
for i in outliers_rms:
	outliers_values.append(rms[i])

plt.figure()
plt.plot(rms)
plt.plot(outliers_rms,outliers_values,"ro", marker = "o")
plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")

plt.savefig('outliers RMS sub1 run1')

