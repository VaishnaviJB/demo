#!/usr/bin/env python
# coding: utf-8

# # Manipulate using a List
# 1.To add new elements to the end of the list
# 2.To reverse elements in the list
# 3.to display the same list of elements multiple times
# 4.To concatenate two list
# 5.To sort the elements in the list in ascending Order
# 

# In[1]:


lst=[7,18,20]
lst.append(28)
lst


# In[2]:


lst=[7,18,20]
lst.reverse()
print(lst)


# In[4]:


lst=[7,18,20]
result=lst*20
print(result)


# In[5]:


lst1=[1,2,3]
lst2=[4,5,6]
concatenated_lst=lst1+lst2
print(concatenated_lst)


# In[6]:


lst=[18,19,20]
lst.sort()
print(lst)


# In[ ]:


2)python program to do in the tuples
1. manipulate using tuples
2.to add new elements to the end of the tuples
3.to reverse elements in the list
4 to display the elements of the same tuple multiple times
5.to concatenate two tuples
6. to sort the elements in the list in ascending order


# In[8]:


etuple =(1,2,3,4,5)
element=6
etuple+=(element,)
reversed_tuple=tuple(reversed(etuple))
n=3
multiplied_tuple=etuple*n
tuple1=(7,8)
tuple2=(9,10)
concatenated_tuple=tuple1+tuple2
sorted_tuple=tuple(sorted(etuple))
print("original tuple:",etuple)
print("reversed tuple:",reversed_tuple)
print("multiplied tuple:",multiplied_tuple)
print("concatenated tuple:",concatenated_tuple)
print("sorted tuple:",sorted_tuple)


# In[ ]:


3. Write a python program to implement the following using list.

Create a list with integers (minimum 10 numbers)

How to display the last number in the list

Command for displaying the values from the list [0:4]

iv) Command for displaying the values from the list [2:]

v) Command for displaying the values from the list [:6]


# In[9]:


# Create a list with integers (minimum 10 numbers)
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Display the last number in the list
last_number = my_list[-1]
print("Last number in the list:", last_number)

# Display values from the list [0:4]
subset1 = my_list[0:4]
print("Values from the list [0:4]:", subset1)

# Display values from the list [2:]
subset2 = my_list[2:]
print("Values from the list [2:]:", subset2)

# Display values from the list [:6]
subset3 = my_list[:6]
print("Values from the list [:6]:", subset3)


# In[ ]:


4. Write a Python program: tuple1 = (10,50,20,40,30)

i. To display the elements 10 and 50 from tuple1

ii. To display the length of a tuple1.

iii. To find the minimum element from tuple1.

iv. To add all elements in the tuple1.

V. To display the same tuple1 multiple times.


# In[10]:


# Define the tuple
tuple1 = (10, 50, 20, 40, 30)

# i. To display the elements 10 and 50 from tuple1
print("Elements 10 and 50 from tuple1:", tuple1[0], tuple1[1])

# ii. To display the length of tuple1
print("Length of tuple1:", len(tuple1))

# iii. To find the minimum element from tuple1
min_element = min(tuple1)
print("Minimum element from tuple1:", min_element)

# iv. To add all elements in tuple1
sum_elements = sum(tuple1)
print("Sum of elements in tuple1:", sum_elements)

# v. To display the same tuple1 multiple times
n = 3  # Number of times to display tuple1
tuple1_multiple_times = tuple1 * n
print("Tuple1 displayed multiple times:", tuple1_multiple_times)


# In[ ]:


5. Write a Python program:

To calculate the length of a string

ii. To reverse words in a string

To display the same string multiple times

iv. To concatenate two strings

V. Str1=" South India", using string slicing to display "India"


# In[11]:


# Calculate the length of a string
str1 = "Hello, World!"
length = len(str1)
print("Length of the string:", length)

# Reverse words in a string
str2 = "This is a sample string"
reversed_words = ' '.join(str2.split()[::-1])
print("Reversed words in the string:", reversed_words)

# Display the same string multiple times
str3 = "Repeat me! "
repeated_string = str3 * 3
print("Repeated string:", repeated_string)

# Concatenate two strings
str4 = "Hello, "
str5 = "Python!"
concatenated_string = str4 + str5
print("Concatenated string:", concatenated_string)

# Using string slicing to display "India" from "South India"
str6 = "South India"
india_part = str6[6:]
print("Sliced 'India' from 'South India':", india_part)


# In[ ]:


6. Perform the following:

1) Creating the Dictionary.

11) Accessing values and keys in the Dictionary.

Updating the dictionary using a function.

iv) Clear and delete the dictionary values.


# In[13]:


my_dict = {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
}
# Accessing values by key
value = my_dict["key1"]
print(value)  # This will print "value1"

# Accessing keys
keys = my_dict.keys()
print(keys)  # This will print dict_keys(['key1', 'key2', 'key3'])
# Define a function to update the dictionary
def update_dict(d, key, value):
    d[key] = value

# Call the function to update the dictionary
update_dict(my_dict, "key4", "value4")
print(my_dict)  # This will include the updated key-value pair
my_dict.clear()
del my_dict["key1"]


# In[14]:


print(my_dict) 
my_dict.clear()
del my_dict["key1"]


# In[ ]:


7. insert anyposition in a list


# In[ ]:


print("Enter 10 Elements of List: ")
nums = []
for i in range(10):
    nums.insert(i, input())
print("Enter an Element to Insert at End: ")
elem = input()
nums.append(elem)
print("\nThe New List is: ")
print(nums)


# In[ ]:


8 . To delete an element from a list by its index in Python


# In[2]:


my_list = [1, 2, 3, 4, 5]
index_to_delete = 2  # Index of the element to delete

if index_to_delete < len(my_list):
    deleted_element = my_list.pop(index_to_delete)
    print("Deleted element:", deleted_element)
else:
    print("Index out of range")

print("Updated list:", my_list)


# In[ ]:


9.display a number from 1 to 100


# In[3]:


import random

random_number = random.randint(1, 100)
print(random_number)


# In[ ]:


11. Create a dictionary containing three lambda functions square, cube and square root.

i) E.g. dict('Square': function for squaring, 'Cube': function for cube, 'Squareroot': function for square root}

ii) Pass the values (input from the user) to the functions in the dictionary respectively.

Then add the outputs of each function and print it.


# In[4]:


# Define the dictionary with lambda functions
func_dict = {
    'Square': lambda x: x**2,
    'Cube': lambda x: x**3,
    'Squareroot': lambda x: x**0.5
}

# Get user input for a number
num = float(input("Enter a number: "))

# Initialize a variable to store the sum of outputs
result = 0

# Iterate through the functions in the dictionary, apply them, and add to the result
for func_name, func in func_dict.items():
    output = func(num)
    result += output

# Print the sum of outputs
print("Sum of outputs:", result)


# In[ ]:


12. 12. A list of words is given. Find the words from the list that have their second character in uppercase. ['hello', 'Dear', 'how', 'ARe', 'You']


# In[26]:


word_list = ['hello', 'Dear', 'how', 'ARe', 'You']
result = []

for word in word_list:
    if len(word) > 1 and word[1].isupper():
        result.append(word)

print(result)


# In[ ]:


13.13. A dictionary of names and their weights on earth is given. Find how much they will weigh on the moon. (Use map and lambda functions) Formula: wMoon = (wEarth GMoon) / GEarth *

#Weight of people in kg

WeightOnEarth = ('John':45, 'Shelly':65, 'Marry':35)

# Gravitational force on the Moon: 1.622 m/s2 GMoon 1.622

# Gravitational force on the Earth: 9.81 m/s2

GEarth = 9.81

2

4()


# In[27]:


# Weight of people in kg
WeightOnEarth = {'John': 45, 'Shelly': 65, 'Marry': 35}

# Gravitational force on the Moon: 1.622 m/s^2
GMoon = 1.622

# Gravitational force on the Earth: 9.81 m/s^2
GEarth = 9.81

# Calculate weight on the Moon for each person
WeightOnMoon = list(map(lambda person: (person[0], (person[1] * GMoon) / GEarth), WeightOnEarth.items()))

# Print the results
for person, weight_moon in WeightOnMoon:
    print(f"{person}'s weight on the Moon: {weight_moon:.2f} kg")


# # CONTROL STRUCTURES

# In[ ]:





# In[ ]:


PROGRAM TO FIND THE FIRST N PRIME NUMBERS


# In[1]:


def is_prime(num):
    if num <= 1:
        return False
    elif num <= 3:
        return True
    elif num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

n = int(input("Enter the value of n: "))
prime_numbers = first_n_primes(n)
print(f"The first {n} prime numbers are: {prime_numbers}")


# In[ ]:


2. Write the python code that calculates the salary of an employee. Prompt the user to enter the Basic Salary, HRA, TA, and DA. Add these components to calculate the Gross Salary. Also, deduct 10% of salary from the Gross Salary to be paid as tax and display gross minus tax as net salary.


# In[2]:


# Prompt the user to enter Basic Salary, HRA, TA, and DA
basic_salary = float(input("Enter Basic Salary: "))
hra = float(input("Enter HRA: "))
ta = float(input("Enter TA: "))
da = float(input("Enter DA: "))

# Calculate Gross Salary by adding Basic, HRA, TA, and DA
gross_salary = basic_salary + hra + ta + da

# Calculate Tax (10% of Gross Salary)
tax = 0.10 * gross_salary

# Calculate Net Salary (Gross Salary - Tax)
net_salary = gross_salary - tax

# Display Gross and Net Salary
print(f"Gross Salary: {gross_salary}")
print(f"Net Salary: {net_salary}")


# In[ ]:


3.search for a string in a given list


# In[3]:


def search_string_in_list(search_string, my_list):
    found_indices = []
    for i, item in enumerate(my_list):
        if search_string in item:
            found_indices.append(i)
    
    if found_indices:
        print(f"'{search_string}' found in the list at indices: {found_indices}")
    else:
        print(f"'{search_string}' not found in the list.")

# Example usage:
my_list = ["apple", "banana", "cherry", "date", "banana"]
search_string = "banana"
search_string_in_list(search_string, my_list)


# In[ ]:


4. Write a Python function that accepts a string and calculates the number of upper-case letters and lower-case letters.


# In[7]:


def count_case_letters(input_string):
    upper_count = 0
    lower_count = 0
    
    for char in input_string:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1
    
    return upper_count, lower_count
input_str = "Hello World"
upper, lower = count_case_letters(input_str)
print("Uppercase letters:", upper)
print("Lowercase letters:", lower)



# In[ ]:


5. Write a program to display the sum of odd numbers and even numbers that fall between 12 and 37.


# In[8]:


# Initialize variables to store the sums
sum_odd = 0
sum_even = 0

# Loop through numbers from 12 to 37
for num in range(12, 38):
    # Check if the number is odd or even
    if num % 2 == 0:
        # If even, add it to the sum_even
        sum_even += num
    else:
        # If odd, add it to the sum_odd
        sum_odd += num

# Display the sums
print("Sum of even numbers:", sum_even)
print("Sum of odd numbers:", sum_odd)


# In[ ]:


6.print the table of any number


# In[9]:


# Get the number from the user
num = int(input("Enter a number: "))

# Print the multiplication table
print(f"Multiplication Table of {num}:")
for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")


# In[ ]:


7.sum the first 10 prime numbers


# In[10]:


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

count = 0
sum_of_primes = 0
number = 2

while count < 10:
    if is_prime(number):
        sum_of_primes += number
        count += 1
    number += 1

print(f"The sum of the first 10 prime numbers is: {sum_of_primes}")


# In[ ]:


8.You can implement arithmetic operations using nested if statements in Python like this:


# In[11]:


# Get two numbers from the user
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

# Get the desired operation from the user
operation = input("Enter an arithmetic operation (+, -, *, /): ")

# Perform the selected operation
if operation == "+":
    result = num1 + num2
elif operation == "-":
    result = num1 - num2
elif operation == "*":
    result = num1 * num2
elif operation == "/":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Division by zero is not allowed."
else:
    result = "Invalid operation"

print(f"Result: {result}")


# In[ ]:


9.The temperature in celsius and convert it to a Fahrenheit


# In[12]:


# Input temperature in Celsius
celsius = float(input("Enter temperature in Celsius: "))

# Convert to Fahrenheit
fahrenheit = (celsius * 9/5) + 32

# Display the result
print(f"{celsius} degrees Celsius is equal to {fahrenheit} degrees Fahrenheit")


# In[ ]:


10.Find a maximum and minimum number in a list without using an inbuilt function


# In[13]:


# Function to find maximum and minimum in a list
def find_max_min(numbers):
    # Check if the list is empty
    if not numbers:
        return None, None

    # Initialize variables to store maximum and minimum
    maximum = minimum = numbers[0]

    # Iterate through the list
    for number in numbers:
        if number > maximum:
            maximum = number
        if number < minimum:
            minimum = number

    return maximum, minimum

# Example usage
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
max_num, min_num = find_max_min(numbers)
print(f"Maximum: {max_num}")
print(f"Minimum: {min_num}")


# In[ ]:


11.Write a program in python to print out the number of seconds in 30-day month 30 days, 24 hours in a day, 60 minutes per day, 60 seconds in a minute.


# In[14]:


days_in_month = 30
hours_in_day = 24
minutes_in_hour = 60
seconds_in_minute = 60

seconds_in_30_days = days_in_month * hours_in_day * minutes_in_hour * seconds_in_minute

print(f"There are {seconds_in_30_days} seconds in a 30-day month.")


# In[ ]:


12.printout the number of seconds in a year


# In[15]:


# Constants for the number of days, hours, minutes, and seconds
days_per_year = 365
hours_per_day = 24
minutes_per_hour = 60
seconds_per_minute = 60

# Calculate the total number of seconds in a year
total_seconds = days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute

# Display the result
print(f"The number of seconds in a year (assuming 365 days) is: {total_seconds} seconds")


# In[ ]:


13. A high-speed train can travel at an average speed of 150 mph, how long will it take a train travelling at this speed to travel from London to Glasgow which is 414 miles


# In[16]:


# Define the distance in miles and the average speed in mph
distance = 414
speed = 150

# Calculate the time in hours
time_hours = distance / speed

# Convert hours to hours and minutes
hours = int(time_hours)
minutes = (time_hours - hours) * 60

# Print the result
print(f"It will take approximately {hours} hours and {minutes:.2f} minutes to travel from London to Glasgow.")


# In[17]:


14. # Define the variable days_in_each_school_year
days_in_each_school_year = 192

# Years 7 to 11
years = range(7, 12)

# Calculate the total hours spent in school
total_hours = sum(year * days_in_each_school_year * 6 for year in years)

# Display the result
print(f"Total hours spent in school from year 7 to year 11: {total_hours} hours")


# In[ ]:


15. If the age of Ram,Sam and Khan are input through the keyboard, write a python program to determine the eldest and youngest of the three


# In[18]:


# Input ages of Ram, Sam, and Khan
ram_age = int(input("Enter Ram's age: "))
sam_age = int(input("Enter Sam's age: "))
khan_age = int(input("Enter Khan's age: "))

# Determine the eldest and youngest
if ram_age >= sam_age and ram_age >= khan_age:
    eldest = "Ram"
    if sam_age <= khan_age:
        youngest = "Sam"
    else:
        youngest = "Khan"
elif sam_age >= ram_age and sam_age >= khan_age:
    eldest = "Sam"
    if ram_age <= khan_age:
        youngest = "Ram"
    else:
        youngest = "Khan"
else:
    eldest = "Khan"
    if ram_age <= sam_age:
        youngest = "Ram"
    else:
        youngest = "Sam"

# Print the results
print(f"The eldest among Ram, Sam, and Khan is: {eldest}")
print(f"The youngest among Ram, Sam, and Khan is: {youngest}")


# In[ ]:


15.with nd without slicing


# In[19]:


def rotate_list_using_slicing(input_list, n):
    if len(input_list) == 0:
        return input_list
    
    n %= len(input_list)  # Ensure n is within the length of the list
    rotated_list = input_list[-n:] + input_list[:-n]
    return rotated_list

# Input list
my_list = [1, 2, 3, 4, 5]
n = int(input("Enter the number of times to rotate to the right: "))

rotated_list = rotate_list_using_slicing(my_list, n)
print("Rotated list using slicing technique:", rotated_list)


# In[20]:


def rotate_list_without_slicing(input_list, n):
    if len(input_list) == 0:
        return input_list
    
    n %= len(input_list)  # Ensure n is within the length of the list
    for _ in range(n):
        temp = input_list.pop()
        input_list.insert(0, temp)
    return input_list

# Input list
my_list = [1, 2, 3, 4, 5]
n = int(input("Enter the number of times to rotate to the right: "))

rotated_list = rotate_list_without_slicing(my_list, n)
print("Rotated list without slicing technique:", rotated_list)


# In[ ]:


16.print the patterns given below


# In[21]:


1. # Input the number of rows for the pattern
n = int(input("Enter the number of rows: "))

# Function to calculate binomial coefficients
def binomial_coefficient(n, k):
    if k == 0 or k == n:
        return 1
    return binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)

# Loop to print the pattern
for i in range(n):
    for j in range(i + 1):
        print(binomial_coefficient(i, j), end=" ")
    print()


# In[ ]:


2.Pattern program 
*
* *
* * *
* * * *
* * * * * 
get_ipython().run_line_magic('pinfo', 'program')


# In[22]:


n = 5  # Number of rows

# Outer loop for rows
for i in range(n):
    # Inner loop for columns
    for j in range(i + 1):
        print("*", end=" ")
    print()  # Move to the next line after each row


# In[23]:


3 # Input the number of rows for the pattern
n = int(input("Enter the number of rows: "))

# Loop to print the pattern
for i in range(1, n + 1):
    print(" " * (n - i), end="")  # Print spaces before asterisks
    print("* " * i)  # Print asterisks with a space in between


# In[25]:


word = "Python"
for i in range(len(word) + 1):
    print(word[:i])
print("Python program?")


# In[ ]:




