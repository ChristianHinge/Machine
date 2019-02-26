import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# In this exercise we will rely on pandas for some of the processing steps:
import pandas as pd
def oneOutOfK(value):
     #One-Out-of-K
     # integer encode
     label_encoder = LabelEncoder()
     integer_encoded = label_encoder.fit_transform(value)
     #print(integer_encoded)
     # binary encode
     onehot_encoder = OneHotEncoder(sparse=False)
     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
     return onehot_encoded

# We start by defining the path to the file that we're we need to load.
file_path = '../Data/car.data'
# First off we simply read the file in using readtable, however, we need to
# tell the function that the file is tab-seperated. We also need to specify
# that there is no header
data = pd.read_csv(file_path, sep=',', header=None)


# We manually type the attribute names
attributeNames = np.asarray(["symboling", "normalized-losses", "make", "fuel-type","aspiration", "num-of-doors", 
                            "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", 
                            "curb-weight", "engine-type", "num-of-cylinders", "engine-size","fuel-system","bore","stroke", 
                            "compression-ratio","horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"])
# We assign the attribute names to the data columns
data.columns=attributeNames

#Changing "?" with "NaN"
for col in attributeNames:
    data[col] = data[col].replace('?','NaN')
    data[col] = data[col].replace('two','2')
    data[col] = data[col].replace('three','3')
    data[col] = data[col].replace('four','4')
    data[col] = data[col].replace('five','5')
    data[col] = data[col].replace('six','6')
    data[col] = data[col].replace('eight','8')
    data[col] = data[col].replace('twelve','12')
    
print(np.array(data["num-of-doors"]))    

# As we progress through this script, we might change which attributes are
# stored where. For simplicity in presenting the processing steps, we wont
# keep track of those changes in attributeNames in this example script.

# The last column is a unique string for each observation defining the
# car make and model. We decide to extract this in a variable for itself
# for now, and then remove it from data:
    
#Remember types of attribute
car_names = np.unique(data["make"])
fuel_types_names= np.unique(data["fuel-type"])
aspiration_types=np.unique(data["aspiration"])
body_style_names=np.unique(data["body-style"])
drive_wheels_types=np.unique(data["drive-wheels"])
engine_location_types=np.unique(data["engine-location"])
engine_type_names=np.unique(data["engine-type"])
fuel_system_types = np.unique(data["fuel-system"])

#One Out of K Columns
make=oneOutOfK(np.array(data["make"]))
fuel_type=oneOutOfK(np.array(data["fuel-type"]))
aspiration=oneOutOfK(np.array(data["aspiration"]))
body_style=oneOutOfK(np.array(data["body-style"]))
drive_wheels=oneOutOfK(np.array(data["drive-wheels"]))
engine_location=oneOutOfK(np.array(data["engine-location"]))
engine_type=oneOutOfK(np.array(data["engine-type"]))
fuel_systems=oneOutOfK(np.array(data["fuel-system"]))

#Delete columns
data = data.drop(['make'],axis=1)
data = data.drop(['fuel-type'],axis=1)
data = data.drop(['aspiration'],axis=1)
data = data.drop(['body-style'],axis=1)
data = data.drop(['drive-wheels'],axis=1)
data = data.drop(['engine-location'],axis=1)
data = data.drop(['engine-type'],axis=1)
data = data.drop(['fuel-system'],axis=1)

#Assemble matrix with new columns
data = np.hstack((data,make, fuel_type, aspiration, body_style,drive_wheels,engine_location,engine_type,fuel_systems))

#Delete attributes from attributeNames
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("make"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("fuel-type"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("aspiration"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("body-style"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("drive-wheels"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("engine-location"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("engine-type"))
attributeNames=np.delete(attributeNames, attributeNames.tolist().index("fuel-system"))

#Add new names to attributeNames

attributeNames=np.hstack((attributeNames,car_names , fuel_types_names, aspiration_types, body_style_names,drive_wheels_types,engine_location_types,engine_type_names,fuel_system_types))
data=np.vstack((attributeNames,data))

# Inspect messy data by e.g.:
#print(data.to_string())



# the data has some zero values that the README.txt tolds us were missing
# values - this was specifically for the attributes mpg and displacement,
# so we're careful only to replace the zeros in these attributes, since a
# zero might be correct for some other variables:
#data.mpg = data.mpg.replace({'0': np.nan})
#data.displacement = data.displacement.replace({'0': np.nan})

# We later on find out that a value of 99 for the mpg is not value that is
# within reason for the MPG of the cars in this dataset. The observations
# that has this value of MPG is therefore incorrect, and we should treat
# the value as missing. How would you add a line of code to this data
# cleanup script to account for this information?

## X,y-format
# If the modelling problem of interest was a classification problem where
# we wanted to classify the origin attribute, we could now identify obtain
# the data in the X,y-format as so:
# =============================================================================
# data = np.array(data.get_values(), dtype=np.float64)
# X_c = data[:, :-1].copy()
# y_c = data[:, -1].copy()
# 
# # However, if the problem of interest was to model the MPG based on the
# # other attributes (a regression problem), then the X,y-format is
# # obtained as:
# X_r = data[:, 1:].copy()
# y_r = data[:, 0].copy()
# 
# # Since origin is categorical variable, we can do as in previos exercises
# # and do a one-out-of-K encoding:
# origin = np.array(X_r[:, -1], dtype=int).T-1
# K = origin.max()+1
# origin_encoding = np.zeros((origin.size, K))
# origin_encoding[np.arange(origin.size), origin] = 1
# X_r = np.concatenate((X_r[:, :-1], origin_encoding),axis=1)
# # Since the README.txt doesn't supply a lot of information about what the
# # levels in the origin variable mean, you'd have to either make an educated
# # guess based on the values in the context, or preferably obtain the
# # information from any papers that might be references in the README.
# # In this case, you can inspect origin and car_names, to see that (north)
# # american makes are all value 0 (try looking at car_names[origin == 0],
# # whereas origin value 1 is European, and value 2 is Asian.
# 
# ## Missing values
# # In the above X,y-matrices, we still have the missing values. In the
# # following we will go through how you could go about handling the missing
# # values before making your X,y-matrices as above.
# 
# # Once we have identified all the missing data, we have to handle it
# # some way. Various apporaches can be used, but it is important
# # to keep it mind to never do any of them blindly. Keep a record of what
# # you do, and consider/discuss how it might affect your modelling.
# 
# # The simplest way of handling missing values is to drop any records 
# # that display them, we do this by first determining where there are
# # missing values:
# missing_idx = np.isnan(data)
# # Observations with missing values have a row-sum in missing_idx
# # which is greater than zero:
# obs_w_missing = np.sum(missing_idx, 1) > 0
# data_drop_missing_obs = data[np.logical_not(obs_w_missing), :]
# # This reduces us to 15 observations of the original 29.
# 
# # Another approach is to first investigate where the missing values are.
# # A quick way to do this is to visually look at the missing_idx:
# plt.title('Visual inspection of missing values')
# plt.imshow(missing_idx)
# plt.ylabel('Observations'); plt.xlabel('Attributes');
# plt.show()
# 
# # From such a plot, we can see that the issue is the third column, the
# # displacement attribute. This can be confirmed by e.g. doing:
# #np.sum(missing_idx, 0)
# # which shows that 12 observations are missing a value in the third column. 
# # Therefore, another way to move forward is to disregard displacement 
# # (for now) and remove the attribute. We then remove the few
# # remaining observations with missing values:
# cols = np.ones((data.shape[1]), dtype=bool)
# cols[2] = 0
# data_wo_displacement = data[:, cols] 
# obs_w_missing_wo_displacement = np.sum(np.isnan(data_wo_displacement),1)>0
# data_drop_disp_then_missing = data[np.logical_not(obs_w_missing_wo_displacement), :]
# # Now we have kept all but two of the observations. This however, doesn't
# # necesarrily mean that this approach is superior to the previous one,
# # since we have now also lost any and all information that we could have
# # gotten from the displacement attribute. 
# 
# # One could impute the missing values - "guess them", in some
# # sense - while trying to minimize the impact of the guess.
# # A simply way of imputing them is to replace the missing values
# # with the median of the attribute. We would have to do this for the
# # missing values for attributes 1 and 3:
# data_imputed = data.copy()
# for att in [0, 2]:
#      # We use nanmedian to ignore the nan values
#     impute_val = np.nanmedian(data[:, att])
#     idx = missing_idx[:, att]
#     data_imputed[idx, att] = impute_val
# =============================================================================

