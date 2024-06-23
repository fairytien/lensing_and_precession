# %%
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def __str__(self):
        return f"{self.make} {self.model} {self.year}"

    def start_engine(self):
        print("Engine started.")

    def stop_engine(self):
        print("Engine stopped.")

# %%
# write a class that inherits from Car
class ElectricCar(Car):
    def __init__(self, make, model, year):
        # call the constructor of the parent class
        super().__init__(make, model, year)

    # override the start_engine method of the parent class
    def start_engine(self):
        print("Engine started. Battery running.")

    # add a new method
    def recharge(self):
        print("Battery charging.")

# %%
class Fruit:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __str__(self):
        return f"{self.color} {self.name}"

    def is_edible(self):
        return True
    
# %%
# write a class that inherits from Fruit
class Apple(Fruit):
    def __init__(self, name, color, variety):
        # call the constructor of the parent class
        super().__init__(name, color)
        self.variety = variety

    # override the is_edible method of the parent class
    def is_edible(self):
        return False

    # add a new method
    def is_delicious(self):
        return True

# %%
def whichClass(Class):
    if isinstance(Class, Car):
        print("This is a Car")
    elif isinstance(Class, ElectricCar):
        print("This is an ElectricCar")
    else:
        print("This is not a Car or an ElectricCar")

# %%
car = Car("Honda", "Civic", 2017)
whichClass(car)

# %%
electricCar = ElectricCar("Tesla", "Model 3", 2019)
whichClass(electricCar)

# %%
def whichClass_naive(Class):
    if Class == Car:
        print("This is a Car")
    elif Class == ElectricCar:
        print("This is an ElectricCar")
    else:
        print("This is not a Car or an ElectricCar")

# %%
whichClass_naive(Car)
whichClass_naive(ElectricCar)
whichClass_naive(Fruit)
whichClass(Fruit)

# %%
print(str(Car))
print(str(ElectricCar))
print(str(Fruit))
print(str(car))
print(str(electricCar))
print(type(Car))
print(type(ElectricCar))
print(type(Fruit))
print(type(car))
print(type(electricCar))

# %%
# print Class name
print(Car.__name__)
print(ElectricCar.__name__)
print(Fruit.__name__)
print(car.__class__.__name__)
print(electricCar.__class__.__name__)

# %%
# define an example dictionary
d = {"a": 1, "b": 2, "c": 3}
def test_global():
    global d
    # print the dictionary
    print(d)
    # modify the dictionary
    d["a"] = 10
    d["b"] = 20
    d["c"] = 30
    # print the dictionary again
    print(d)

test_global()

# %%
# define a global variable
x = 10

# define a function that modifies the global variable
def modify_global():
    global x, d
    x = 20
    d["a"] = 100
    d["b"] = 200

# print the value of x before and after calling the function
print(x)  # output: 10
modify_global()
print(x)  # output: 20
print(d)

# %%
import numpy as np

# Original array
a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)  # Output: (6,)
print(a)

# Reshape to 2D array with one row
b = a.reshape(1, -1)
print(b.shape)  # Output: (1, 6)
print(b)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a sample dataset
np.random.seed(0)
category = np.repeat(['Category 1', 'Category 2'], 100)
value = np.concatenate([np.random.normal(loc=5, scale=2, size=100), 
                        np.random.normal(loc=10, scale=3, size=100)])
df = pd.DataFrame({'Category': category, 'Value': value})

# Create the violin plot
sns.violinplot(x='Category', y='Value', data=df)

# Show the plot
plt.show()

# %%
import numpy as np

# Set the number of iterations for the Monte Carlo simulation
iterations = 1000000

# Generate random points in the square with sides of length 1
points = np.random.rand(iterations, 2)

# Determine the number of points in the quarter circle
# This is done by checking if the point's distance from the origin is less than 1
inside_circle = np.sum(np.square(points).sum(axis=1) <= 1)

# The ratio of the points inside the circle to the total points should be pi / 4
pi_estimate = 4 * inside_circle / iterations

print(f"Estimated value of pi: {pi_estimate}")

# %%
# Define the states
states = ["sunny", "rainy"]

# Define the transition matrix
transition_matrix = np.array([[0.8, 0.2],  # Probabilities of going from sunny to sunny or rainy
                              [0.6, 0.4]]) # Probabilities of going from rainy to sunny or rainy

# Start with a sunny day
weather = [states[0]]

# Simulate 10 days of weather
for _ in range(10):
    if weather[-1] == "sunny":
        change = np.random.choice(transition_matrix[0], replace=True)
        if change < 0.8:
            weather.append("sunny")
        else:
            weather.append("rainy")
    elif weather[-1] == "rainy":
        change = np.random.choice(transition_matrix[1], replace=True)
        if change < 0.6:
            weather.append("sunny")
        else:
            weather.append("rainy")

print(weather)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create some data
time_step = 0.02
period = 5.
time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + 0.5 * np.random.randn(time_vec.size)

# Step 3: Perform the Fourier transform using numpy
fft_vals = np.fft.fft(sig)

# Get the absolute value of the complex numbers
fft_abs = np.abs(fft_vals)

# Step 4: Visualize the result
plt.figure()
plt.plot(fft_abs)
plt.title('Fourier transform')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Sample data (time-domain signal)
# Replace this with your own data
t = np.linspace(0, 1, 1000, endpoint=False)  # time points
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)  # Example signal

# Perform Fourier transform
freq = np.fft.fftfreq(len(t), t[1] - t[0])  # Frequency values
fft_result = np.fft.fft(signal)  # Fourier transform

# Plot the time-domain signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Time-Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot the frequency-domain representation
plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(fft_result))
plt.title('Frequency-Domain Representation')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# %%
# Shift zero frequency component to the center
freq_shifted = np.fft.fftshift(freq)
fft_result_shifted = np.fft.fftshift(fft_result)

# Plot the shifted frequency-domain representation
plt.plot(freq_shifted, np.abs(fft_result_shifted))
plt.title('Shifted Frequency-Domain Representation')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# %%
import sympy as sp

# Define symbols
t, s = sp.symbols('t s', real=True)
m, c, k = sp.symbols('m c k', positive=True, real=True)
x = sp.Function('x')(t)
F = sp.Function('F')(t)

# Define the differential equation
diff_eq = m*x.diff(t, t) + c*x.diff(t) + k*x - F

# Assume a specific form for the external force F(t)
F_t = sp.sin(t)

# Apply Laplace transform to both sides of the differential equation
laplace_eq = sp.laplace_transform(diff_eq, t, s, noconds=True)
laplace_F = sp.laplace_transform(F_t, t, s, noconds=True)

# Solve for X(s) (the Laplace transform of x(t))
X_s = sp.solve(laplace_eq, sp.laplace_transform(x, t, s, noconds=True))[0]

# Substitute the external force Laplace transform into the solution
X_s_with_F = X_s.subs(sp.laplace_transform(F, t, s, noconds=True), laplace_F)

# Print the Laplace transform of the solution with the external force
print('X(s) with F(t) =', X_s_with_F)

# %%
