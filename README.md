# shift-scheduler

Creating a full-featured shift scheduler with machine learning to optimize scheduling based on employee availability, preferences, and labor laws is quite an extensive task. Below is a simplified Python program that covers the essential components of such a tool. A real-world application would require a deeper integration with databases, a more advanced machine learning model, thorough testing, and domain-specific compliance checking.

We'll use libraries like `pandas` for data manipulation and `sklearn` for a basic machine learning approach. Here's a starting point:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data structure
data = {
    'employee_id': [1, 2, 3, 4, 5],
    'availability_monday': [8, 4, 8, 8, 4],
    'availability_tuesday': [4, 4, 8, 8, 4],
    'availability_wednesday': [8, 8, 8, 4, 4],
    'preference_morning': [1, 0, 1, 0, 1],
    'preference_afternoon': [0, 1, 0, 1, 0],
    'preference_night': [0, 0, 0, 0, 0],
    'hours_per_week': [20, 20, 20, 20, 20],
    'current_schedule_hours': [10, 12, 16, 14, 10],
}

# Convert data into a dataframe
df = pd.DataFrame(data)

# Define the features and labels
X = df.drop(columns=['employee_id'])
y = (df['hours_per_week'] - df['current_schedule_hours']) > 0

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple model for demonstration purposes
model = RandomForestClassifier()

# Error handling for model fitting
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Predict which employees can be scheduled more shifts
try:
    predictions = model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")

# Display the results
try:
    results = pd.DataFrame({
        'employee_id': df.loc[y_test.index, 'employee_id'],
        'can_be_scheduled_more': predictions
    })

    print(results)
except Exception as e:
    print(f"Error during results generation: {e}")
```

### Key Features in the Program:

- **Data Handling**: The employee data is structured in a pandas DataFrame, including employee availability, preferences, and working hours.
  
- **Machine Learning Introduction**: We use a RandomForestClassifier to predict whether an employee can be scheduled more shifts based on their current schedule and availability.
  
- **Basic Error Handling**: We include try-except blocks to catch and handle any exceptions that may occur during model training, prediction, and result generation.

### Notes:

- **Machine Learning**: This example uses a simple RandomForestClassifier for demonstration. A production-level tool would use advanced models and potentially custom algorithms to optimize across many complex constraints.
  
- **Legal Requirements**: This program doesn't enforce labor laws directly; however, real applications must integrate checks for compliance with relevant legislation.
  
- **Scalability and Data Sources**: In a real application, data would come from more extensive, dynamic sources, possibly requiring database interactions or API integrations.

- **User Interface**: A production application would likely include a web or desktop interface for data input, schedule viewing, and modifications by managers.

- **Advanced Features**: Depending on the requirements, more sophisticated scheduling algorithms such as linear programming or combinatorial optimization might be used alongside machine learning techniques.