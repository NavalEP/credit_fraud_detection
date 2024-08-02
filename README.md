
 # Credit Card Fraud Detection

This project aims to detect fraudulent transactions using machine learning techniques. The dataset includes transaction details and personal information, which are preprocessed and used to train a Support Vector Machine (SVM) classifier.

## Project Structure

- `fraudTrain.csv`: Training dataset.
- `fraudTest.csv`: Test dataset.
- `credit_card_fraud_detection.py`: Python script for preprocessing, training, and testing the model.

## Setup and Installation

1. Clone the repository to your local machine.
2. Ensure you have the required Python libraries installed. You can install them using the following command:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn

## Data Preprocessing

1. **Load the Data:**
   - Load the training and test datasets from CSV files.
   - Convert the `trans_date_trans_time` and `dob` columns to datetime format for better handling of dates.

2. **Drop Unnecessary Columns:**
   - Remove columns that are not needed for the analysis or modeling, including `Unnamed: 0`, `cc_num`, `first`, `last`, `street`, `city`, `state`, `zip`, `dob`, `trans_num`, `trans_date_trans_time`.

3. **Handle Missing Values:**
   - Drop all rows containing missing values to ensure clean data for training and testing.

4. **Encode Categorical Variables:**
   - Use `LabelEncoder` to convert categorical columns (`merchant`, `category`, `gender`, `job`) into numerical values suitable for machine learning algorithms.

## Exploratory Data Analysis (EDA)

- Plot the distribution of the target variable `is_fraud` using a pie chart to visualize the proportion of fraudulent vs. non-fraudulent transactions.

## Model Training

1. **Prepare the Data:**
   - Split the data into features (`X`) and the target variable (`Y`). 

2. **Train the Model:**
   - Train a Support Vector Machine (SVM) classifier on the training data.

3. **Evaluate the Model:**
   - Compute the score of the model on the training data to assess its performance.

## Model Testing

1. **Preprocess the Test Data:**
   - Similar to the training data, drop unnecessary columns and encode categorical variables in the test dataset.

2. **Predict and Evaluate:**
   - Use the trained model to predict the `is_fraud` values on the test data.
   - Calculate and print the accuracy of the model on the test dataset.

## Results

- The model's accuracy on the test data is printed at the end of the script. This provides a measure of how well the model is able to identify fraudulent transactions in the test dataset.

## Usage

1. **Run the Script:**
   ```bash
   python credit_card_fraud_detection.py
