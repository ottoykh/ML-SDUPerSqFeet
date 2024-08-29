# ML-SDUPerSqFeet
Machine Learning for the Subdivided Unit Classification with Open Data (based on web crawling per square feet data) 

### Background 
ML can adopted to understanding the factors leading to property subdivision. Those factor are the (1) Aging Properties and High Maintenance costs, (2) Difficulty in Renting Larger Units and (3) Economic Value of Subdivision.

* Older properties often require significant maintenance, making them less appealing to potential renters or buyers.
* Larger properties may be harder to rent out, especially in a market dominated by smaller families. The size of these units can make them unaffordable for many, leading to prolonged vacancies.
* Subdividing a property generally increases its economic value compared to keeping it as a single unit. For property owners, the potential rental income from subdivided units can be significantly higher—often two to three times the rent of the non-subdivided unit when calculated on a per-square-foot basis.

#### EDA about the PerSqFeet to classified the SDU in Hong Kong (Trainning dataset, 12000+ sample)
![Trainning data EDA](https://raw.githubusercontent.com/ottoykh/ML-SDUPerSqFeet/main/Image_Ref/Train_EDA.png "Trainning data EDA").

### Prerequisite
1. Web crawling from the Real-estate online agent 
2. Gather and geocode the data into 'PreSqFeet' and georeference it 
3. Gather the building footprint and associated data from CSDI, such as Building ages, MTR Exit from the iB1000 base map geodatabase (to do the service network analysis for the walking distance up to 5 km per 0.1 km interval), Median household income from C&S in 2016, Management status of the building, Potential estimated subdivided unit aggerated area, Building OP year, Building Storey and Lift availability. 
4. Spatial join (intersect, point to polygon) the web crawled data with the above-mentioned data
5. Export the points into a table in CSV format (utf-8 encoding)
6. Follow the demo (to maintain the same field name)

### Variables for ML classifier
1. Categorical variables
* Lift (does the building have lift? 0,1)
* Observed and defined SDU Protential areas (Polygon, within ? 0,1) 
* With building management communities 業主立案法團 (三無大廈? 0,1)

2. Continuous variables (numerical) 
* Per-Square Feet data from the online agent 
* Building OP year/date 
* Building storey 
* Median household income by street block
* Distance to the nearest MTR exit 

### Demo to use the ML-Model
Git Clone the data from this page 
```
!git clone https://github.com/ottoykh/ML-SDUPerSqFeet.git
```
Import, install packages and unzip the ML model 
```
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import zipfile; zipfile.ZipFile('/content/ML-SDUPerSqFeet/Model/model.zip').extractall('/content/ML-SDUPerSqFeet/Model/')
```
Load the Pre-trained ML model 
```
loaded_clf = joblib.load('/content/ML-SDUPerSqFeet/Model/model.joblib')
le = joblib.load('/content/ML-SDUPerSqFeet/Model/label_encoder.joblib')
print("Model loaded.")
```
Do the prediction for a new data, the following is just a sample: 
```
new_data = pd.read_csv('/content/ML-SDUPerSqFeet/Try/Trail1.csv')
new_X_cont = new_data[['PerSqFeet', 'Year', 'Income2016', 'WalkDis', 'BD_STOREY']]
new_X_cat = new_data[['Lift', 'Potential', 'Management']]
new_X_cat = new_X_cat.apply(le.transform)
new_X = pd.concat([new_X_cont, new_X_cat], axis=1)

y_pred_loaded = loaded_clf.predict(new_X)

loaded_accuracy = accuracy_score(new_data['SDU_cat'], y_pred_loaded)  # Ensure 'SDU_cat' is the true label
print(f'Loaded model accuracy: {loaded_accuracy:.2f}')
print('Loaded model Confusion Matrix:')
print(confusion_matrix(new_data['SDU_cat'], y_pred_loaded))
print('\nLoaded model Classification Report:')
print(classification_report(new_data['SDU_cat'], y_pred_loaded))
```

### Open data can be adopted
* Lift (does the building have lift? 0,1)

Ref: https://portal.csdi.gov.hk/geoportal/?datasetId=emsd_rcd_1638867755735_63142 

* Observed and defined SDU Potential areas (Polygon, within ? 0,1) 

* With building management communities (三無大廈? 0,1)

Ref : https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=landreg_rcd_1668151937201_80478
Potential
* Per-Square Feet data from the online agent 

Ref : https://hk.centanet.com/findproperty/list/buy

Ref : https://www.28hse.com/

Ref : https://www.house730.com/buy/t1/ 

* Building OP year/date 

Ref : https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=bd_rcd_1631167534872_19764

* Building storey 

Ref : https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=landsd_rcd_1637211194312_35158

* Median household income by street block

Ref (2016) : https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=censtatd_rcd_1629267205229_49850

Ref (2021) : https://portal.csdi.gov.hk/geoportal/?lang=zh-hk&datasetId=censtatd_rcd_1635933282224_58228

* Distance to the nearest MTR exit 

Ref : https://raw.githubusercontent.com/ottoykh/LandsD-iB1000/main/Features/RailwayEntrance.geojson


