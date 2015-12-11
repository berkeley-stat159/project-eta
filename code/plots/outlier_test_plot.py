from utils.outliers import *
from utils.load_data import *
from utils.diagnostics import *


#Plots

#subject 1 fMRI data 
data = get_image(3,9).get_data()
std = vol_std(data)
plt.plot(std)



outliers = iqr_outliers(std)[0]
ranges = iqr_outliers(std)[1]

plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")



# RMS outliers plot 

rms = vol_rms_diff(data)



outliers_rms = iqr_outliers(rms)[0]
ranges = iqr_outliers(rms)[1]

outliers_values = []
for i in outliers_rms:
	outliers_values.append(rms[i])


plt.plot(rms)
plt.plot(outliers_rms,outliers_values,"ro", marker = "o")
plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")


#extended difference outliers 

rms.append(0)

extended_outliers = extend_diff_outliers(outliers_rms)

outliers_values_extended = []
for i in extended_outliers:
	outliers_values_extended.append(rms[i])


plt.plot(rms)
plt.plot(extended_outliers,outliers_values_extended,"ro", marker = "o")

plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")




