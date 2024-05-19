#Simharaju Rajkumar DataScience Internship by CodeAlpha
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load and preprocess the dataset (assuming it's in the same directory)
data = pd.read_csv('titanic.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data.fillna(0, inplace=True)

# Select features and target
features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
target = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# GUI
def predict_survival():
    # Retrieve values from the GUI
    pclass = int(entry_pclass.get())
    sex = 0 if var_gender.get() == "Male" else 1
    age = int(entry_age.get())
    sibsp = int(entry_sibsp.get())
    parch = int(entry_parch.get())
    fare = float(entry_fare.get())
    embarked = {'S': 0, 'C': 1, 'Q': 2}[entry_embarked.get().upper()]

    # Create a DataFrame for the new passenger
    new_passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Make a prediction
    prediction = model.predict(new_passenger)
    result = 'Survived' if prediction[0] == 1 else 'Did not survive'
    
    # Show the result in a messagebox
    messagebox.showinfo("Prediction", result)
    
    # Update the pie chart with the new prediction
    update_pie_chart(prediction[0])

def update_pie_chart(survival):
    # Clear the previous chart
    ax.clear()
    
    # Data to plot
    categories = ['Did not survive', 'Survived']
    values = [1 - survival, survival]
    
    # Plot
    ax.bar(categories, values, color=['red','green'])
    ax.set_title('Survival Prediction')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)  # Set the limit of y-axis to match probability range
    canvas.draw()


# Create the main window
root = tk.Tk()
root.title("Titanic Survival Prediction")
root.configure(bg='light blue')

# Create and place input fields
tk.Label(root, text="Passenger Class (1/2/3):").grid(row=0, column=0)
entry_pclass = tk.Entry(root)
entry_pclass.grid(row=0, column=1)

tk.Label(root, text="Gender:").grid(row=1, column=0)
var_gender = tk.StringVar(value="Male")
tk.Radiobutton(root, text="Male", variable=var_gender, value="Male").grid(row=1, column=1)
tk.Radiobutton(root, text="Female", variable=var_gender, value="Female").grid(row=1, column=2)

tk.Label(root, text="Age of the Passenger:").grid(row=2, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=2, column=1)

tk.Label(root, text="Siblings/Spouses Aboard:").grid(row=3, column=0)
entry_sibsp = tk.Entry(root)
entry_sibsp.grid(row=3, column=1)

tk.Label(root, text="Parents/Children Aboard:").grid(row=4, column=0)
entry_parch = tk.Entry(root)
entry_parch.grid(row=4, column=1)

tk.Label(root, text="Fare (Money paid by the Passenger for the Journey) :").grid(row=5, column=0)
entry_fare = tk.Entry(root)
entry_fare.grid(row=5, column=1)

tk.Label(root, text="Embarked (S/C/Q):").grid(row=6, column=0)
entry_embarked = tk.Entry(root)
entry_embarked.grid(row=6, column=1)

# Create and place the prediction button
predict_button = tk.Button(root, text="Predict Survival", command=predict_survival)
predict_button.grid(row=7, column=0, columnspan=2)

# Create a figure for the pie chart
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.get_tk_widget().grid(row=8, column=0, columnspan=2)

# Set colors for the input fields and labels
entry_pclass.config(bg='orange')
entry_age.config(bg='light green')
entry_sibsp.config(bg='light yellow')
entry_parch.config(bg='light pink')
entry_fare.config(bg='light grey')
entry_embarked.config(bg='light coral')

# Set colors for the labels
for label in root.grid_slaves():
    if isinstance(label, tk.Label):
        label.config(bg='light goldenrod yellow')

# Run the application
root.mainloop()  