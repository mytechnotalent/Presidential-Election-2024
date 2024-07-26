# Presidential Election Favorability 2024 

## Overview
This notebook uses classification modeling to predict the likelihood of the next President being Kamala Harris or Donald Trump based on current favorability ratings.

## Objectives
- Predict whether the politician is Kamala Harris or Donald Trump.
- Calculate the accuracy, precision, recall, and f1-score of the model/s.
- Visualize the results using matplotlib.

## Tools Used
- numpy
- pandas
- scikit-learn
- tensorflow
- imblearn
- matplotlib

## Dataset
This dataset provides detailed favorability ratings for both candidates. For this analysis, we focus on the following.
- Politician (Kamala Harris or Donald Trump)
- Favorability

## Model
We will use the following models for this task.
- TensorFlow Sequential

## Credits
**Dataset Author:**
* FiveThirtyEight

**Model Author:**  
* Kevin Thomas

**Date:**  
* 07-26-24  

**Version:**  
* 1.0


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, LeakyReLU # type: ignore
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
```

## Step 1: Data Preparation


```python
# Load the dataset
data = pd.read_csv('favorability_polls_07-26-24.csv')

# Filter the dataset to include only Kamala Harris and Donald Trump
filtered_data = data[(data['politician'] == 'Kamala Harris') | (data['politician'] == 'Donald Trump')]

# Fill NaN values with the mean of the column in a new DataFrame
filtered_data = filtered_data.copy()  # Create a copy of the DataFrame to avoid chained assignment
filtered_data['favorable'] = filtered_data['favorable'].fillna(filtered_data['favorable'].mean())
filtered_data['unfavorable'] = filtered_data['unfavorable'].fillna(filtered_data['unfavorable'].mean())

# Separate the majority and minority classes
harris_data = filtered_data[filtered_data['politician'] == 'Kamala Harris']
trump_data = filtered_data[filtered_data['politician'] == 'Donald Trump']

# Ensure equal number of samples for both candidates
min_samples = min(harris_data.shape[0], trump_data.shape[0])
harris_resampled = resample(harris_data, replace=False, n_samples=min_samples, random_state=42)
trump_resampled = resample(trump_data, replace=False, n_samples=min_samples, random_state=42)

# Combine the resampled datasets
balanced_data = pd.concat([harris_resampled, trump_resampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Prepare data for modeling
X_balanced = balanced_data[['favorable']].values
y_balanced = balanced_data['politician'].values
```

## Step 2: Feature Engineering


```python
# One-hot encode the politician column
encoder = OneHotEncoder(sparse_output=False)
y_balanced_encoded = encoder.fit_transform(balanced_data[['politician']])

# Split data into training and testing sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced_encoded, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_balanced_scaled = scaler.transform(X_test_balanced)

# Calculate class weights to address imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_balanced_encoded, axis=1)),
    y=np.argmax(y_balanced_encoded, axis=1)
)
class_weight_dict = dict(enumerate(class_weights))

# Print class weights
print("Class weights:", class_weight_dict)
```

    Class weights: {0: 1.0, 1: 1.0}


## Step 3: Modeling


```python
# Build the model
model_balanced = Sequential([
    Input(shape=(1,)),
    Dense(16),
    LeakyReLU(alpha=0.01),
    Dense(8),
    LeakyReLU(alpha=0.01),
    Dense(2, activation='softmax')
])

# Compile the model
model_balanced.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the sampling strategy to control the amount of resampling
sampling_strategy = 'minority'  

# Apply SMOTE to balance the training data with adjustable parameters
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=4,  # Adjust the number of neighbors for SMOTE
    random_state=42
)
X_train_balanced_smote, y_train_balanced_smote = smote.fit_resample(X_train_balanced_scaled, np.argmax(y_train_balanced, axis=1))

# Convert y_train_balanced_smote to one-hot encoding
y_train_balanced_smote_encoded = tf.keras.utils.to_categorical(y_train_balanced_smote)

# Train the model with the resampled data
history_balanced = model_balanced.fit(
    X_train_balanced_smote, y_train_balanced_smote_encoded,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_balanced_scaled, y_test_balanced),
    class_weight=class_weight_dict
)

# Evaluate the model
test_loss_balanced, test_accuracy_balanced = model_balanced.evaluate(X_test_balanced_scaled, y_test_balanced)
print(f'Test Accuracy with Balanced Data: {test_accuracy_balanced:.4f}')
```

    Epoch 1/50


    /Users/kevinthomas/Documents/data-science/venv/lib/python3.11/site-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
      warnings.warn(


    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.5222 - loss: 0.6901 - val_accuracy: 0.6038 - val_loss: 0.6808
    Epoch 2/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - accuracy: 0.5784 - loss: 0.6817 - val_accuracy: 0.5974 - val_loss: 0.6797
    Epoch 3/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step - accuracy: 0.5607 - loss: 0.6844 - val_accuracy: 0.5942 - val_loss: 0.6780
    Epoch 4/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 457us/step - accuracy: 0.5853 - loss: 0.6830 - val_accuracy: 0.5942 - val_loss: 0.6756
    Epoch 5/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 453us/step - accuracy: 0.5853 - loss: 0.6796 - val_accuracy: 0.5942 - val_loss: 0.6744
    Epoch 6/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 464us/step - accuracy: 0.5946 - loss: 0.6775 - val_accuracy: 0.5942 - val_loss: 0.6735
    Epoch 7/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 456us/step - accuracy: 0.5993 - loss: 0.6751 - val_accuracy: 0.5751 - val_loss: 0.6739
    Epoch 8/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step - accuracy: 0.5773 - loss: 0.6782 - val_accuracy: 0.5751 - val_loss: 0.6735
    Epoch 9/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step - accuracy: 0.5998 - loss: 0.6759 - val_accuracy: 0.5974 - val_loss: 0.6701
    Epoch 10/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step - accuracy: 0.5748 - loss: 0.6775 - val_accuracy: 0.5942 - val_loss: 0.6717
    Epoch 11/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 454us/step - accuracy: 0.5860 - loss: 0.6728 - val_accuracy: 0.5751 - val_loss: 0.6722
    Epoch 12/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step - accuracy: 0.5963 - loss: 0.6770 - val_accuracy: 0.5751 - val_loss: 0.6737
    Epoch 13/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step - accuracy: 0.5814 - loss: 0.6770 - val_accuracy: 0.5942 - val_loss: 0.6705
    Epoch 14/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step - accuracy: 0.5802 - loss: 0.6739 - val_accuracy: 0.5751 - val_loss: 0.6719
    Epoch 15/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step - accuracy: 0.5879 - loss: 0.6740 - val_accuracy: 0.5751 - val_loss: 0.6709
    Epoch 16/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 461us/step - accuracy: 0.5927 - loss: 0.6715 - val_accuracy: 0.5942 - val_loss: 0.6694
    Epoch 17/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 480us/step - accuracy: 0.6014 - loss: 0.6684 - val_accuracy: 0.5751 - val_loss: 0.6716
    Epoch 18/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 461us/step - accuracy: 0.5734 - loss: 0.6786 - val_accuracy: 0.5751 - val_loss: 0.6705
    Epoch 19/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 472us/step - accuracy: 0.5923 - loss: 0.6765 - val_accuracy: 0.5942 - val_loss: 0.6698
    Epoch 20/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 479us/step - accuracy: 0.6033 - loss: 0.6680 - val_accuracy: 0.5751 - val_loss: 0.6708
    Epoch 21/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 479us/step - accuracy: 0.5601 - loss: 0.6877 - val_accuracy: 0.5751 - val_loss: 0.6711
    Epoch 22/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 472us/step - accuracy: 0.5682 - loss: 0.6793 - val_accuracy: 0.5751 - val_loss: 0.6728
    Epoch 23/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 456us/step - accuracy: 0.5920 - loss: 0.6735 - val_accuracy: 0.5751 - val_loss: 0.6718
    Epoch 24/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 449us/step - accuracy: 0.5811 - loss: 0.6751 - val_accuracy: 0.5751 - val_loss: 0.6717
    Epoch 25/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 455us/step - accuracy: 0.5866 - loss: 0.6749 - val_accuracy: 0.5751 - val_loss: 0.6728
    Epoch 26/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step - accuracy: 0.5815 - loss: 0.6764 - val_accuracy: 0.5751 - val_loss: 0.6714
    Epoch 27/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 450us/step - accuracy: 0.5657 - loss: 0.6787 - val_accuracy: 0.5942 - val_loss: 0.6696
    Epoch 28/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 455us/step - accuracy: 0.5879 - loss: 0.6711 - val_accuracy: 0.5751 - val_loss: 0.6695
    Epoch 29/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 449us/step - accuracy: 0.5891 - loss: 0.6813 - val_accuracy: 0.5751 - val_loss: 0.6705
    Epoch 30/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 459us/step - accuracy: 0.5852 - loss: 0.6753 - val_accuracy: 0.5751 - val_loss: 0.6697
    Epoch 31/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 446us/step - accuracy: 0.5856 - loss: 0.6768 - val_accuracy: 0.5751 - val_loss: 0.6701
    Epoch 32/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 449us/step - accuracy: 0.5878 - loss: 0.6699 - val_accuracy: 0.5751 - val_loss: 0.6726
    Epoch 33/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step - accuracy: 0.5639 - loss: 0.6753 - val_accuracy: 0.5751 - val_loss: 0.6704
    Epoch 34/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step - accuracy: 0.6064 - loss: 0.6691 - val_accuracy: 0.5751 - val_loss: 0.6725
    Epoch 35/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 455us/step - accuracy: 0.5892 - loss: 0.6756 - val_accuracy: 0.5751 - val_loss: 0.6708
    Epoch 36/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 454us/step - accuracy: 0.5739 - loss: 0.6773 - val_accuracy: 0.5751 - val_loss: 0.6722
    Epoch 37/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 451us/step - accuracy: 0.5973 - loss: 0.6700 - val_accuracy: 0.5942 - val_loss: 0.6689
    Epoch 38/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step - accuracy: 0.5978 - loss: 0.6700 - val_accuracy: 0.5751 - val_loss: 0.6736
    Epoch 39/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step - accuracy: 0.5905 - loss: 0.6738 - val_accuracy: 0.5751 - val_loss: 0.6723
    Epoch 40/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 449us/step - accuracy: 0.5871 - loss: 0.6725 - val_accuracy: 0.5751 - val_loss: 0.6715
    Epoch 41/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step - accuracy: 0.5908 - loss: 0.6751 - val_accuracy: 0.5751 - val_loss: 0.6699
    Epoch 42/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 455us/step - accuracy: 0.5680 - loss: 0.6784 - val_accuracy: 0.5751 - val_loss: 0.6704
    Epoch 43/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step - accuracy: 0.5810 - loss: 0.6760 - val_accuracy: 0.5751 - val_loss: 0.6703
    Epoch 44/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 441us/step - accuracy: 0.5806 - loss: 0.6747 - val_accuracy: 0.5751 - val_loss: 0.6721
    Epoch 45/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step - accuracy: 0.6045 - loss: 0.6667 - val_accuracy: 0.5751 - val_loss: 0.6702
    Epoch 46/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 450us/step - accuracy: 0.5925 - loss: 0.6678 - val_accuracy: 0.5751 - val_loss: 0.6727
    Epoch 47/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step - accuracy: 0.5551 - loss: 0.6832 - val_accuracy: 0.5751 - val_loss: 0.6715
    Epoch 48/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 450us/step - accuracy: 0.5831 - loss: 0.6726 - val_accuracy: 0.5942 - val_loss: 0.6691
    Epoch 49/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439us/step - accuracy: 0.5836 - loss: 0.6764 - val_accuracy: 0.5751 - val_loss: 0.6702
    Epoch 50/50
    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 446us/step - accuracy: 0.5782 - loss: 0.6730 - val_accuracy: 0.5751 - val_loss: 0.6716
    [1m10/10[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 262us/step - accuracy: 0.5524 - loss: 0.6786
    Test Accuracy with Balanced Data: 0.5751


## Step 4: Visualization


```python
# Predict on the test set
y_pred_balanced = model_balanced.predict(X_test_balanced_scaled)

# Replace NaN values with zero (if any)
y_pred_balanced = np.nan_to_num(y_pred_balanced)

# Decode the one-hot encoded predictions
y_pred_balanced_decoded = encoder.inverse_transform(y_pred_balanced)

# Calculate the number of times each politician was predicted
harris_count_balanced = (y_pred_balanced_decoded == 'Kamala Harris').sum()
trump_count_balanced = (y_pred_balanced_decoded == 'Donald Trump').sum()

# Plot the bar plot for balanced predictions
fig, ax = plt.subplots()
politicians = ['Kamala Harris', 'Donald Trump']
counts = [harris_count_balanced, trump_count_balanced]
bars = ax.bar(politicians, counts, color=['blue', 'red'])

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

# Add title and labels
ax.set_title('Unfavorable Predictions Harris & Trump (Balanced Data)')
ax.set_ylabel('Number of Predictions')
ax.set_xlabel('Politician')
plt.show()

# Print the model summary
model_balanced.summary()

# Print the model accuracy
print(f'Test Accuracy with Balanced Data: {test_accuracy_balanced:.4f}')
```

    [1m10/10[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 



    
![png](presidential_election_favorability_2024_v10_07_26_24_files/presidential_election_favorability_2024_v10_07_26_24_10_1.png)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">32</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">136</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">18</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">560</span> (2.19 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">186</span> (744.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">374</span> (1.46 KB)
</pre>



    Test Accuracy with Balanced Data: 0.5751


## Step 5: Inference


```python
# Determine the ultimate winner
ultimate_winner_balanced = 'Kamala Harris' if harris_count_balanced > trump_count_balanced else 'Donald Trump'

# Print results
print(f'Kamala Harris predictions: {harris_count_balanced}')
print(f'Donald Trump predictions: {trump_count_balanced}')
print(f'Ultimate Winner: {ultimate_winner_balanced}')
```

    Kamala Harris predictions: 153
    Donald Trump predictions: 160
    Ultimate Winner: Donald Trump

