# Clustering-Electrofacies
A clustering script for python using well log data to classify rock types using k-means clustering.

## The Results

The code takes the assigned well data and generates classifications. This can be used to identify regions of high porosity and therefore the highest production.

The image below shows the classification displayed on a 3D plot where the user can select the data and it will be be displayed on the depth plot showing where the points are found in the well. 

<p align="center">
  <img src="https://github.com/Preston5789/Clustering-Electrofacies/blob/master/Demo_Pic.PNG" width="750" title="hover text">
</p>

## The Code

We first import our data and assign the desired curves.
```
xls_file = pd.ExcelFile("PATH TO YOUR FILE")
df = xls_file.parse('Sheet1')
curves = ['BFV','FFI', 'SGR', 'CMRP_3MS']
```
Then we create a boolean array of the non-null values and then use that to standardize the curves from 0-1 with a standard deviation of 1. 
```
not_null_rows = pd.notnull(df[curves]).any(axis = 1)
X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])
```
We then create a Principal Component Analysis fit find the relevant variables.  
```
#PCA fit creates
pc = PCA(n_components = n_components).fit(X)
```
The transform applies the mapping (transform) to the pca fit.

```
components = pd.DataFrame(data = pc.transform(X), index = df[not_null_rows].index)

```
We then convert components to a matrix and apply the k-means clustering. 
```
#Convert to matrix from minibatch
minibatch_input = components.as_matrix()

##Creates new column and assigns it to MiniBatch Cluster
df.loc[not_null_rows, curve_name] = \
                MiniBatchKMeans(n_clusters = n_clusters,
                batch_size = 100).fit_predict(minibatch_input)
```

## Authors

* **Preston Phillips** - [Preston5789](https://github.com/Preston5789)
