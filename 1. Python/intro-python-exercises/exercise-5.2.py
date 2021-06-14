# Write and test three functions that return the largest, the smallest, and the
# number of dividables by 3 in a given collection of numbers.

def main(numbers):
    """
    a = [2, 4, 6, 12, 15, 99, 100]
    100
    2
    4
    """
    print(str(largest(numbers)) + " is the largest")
    print(str(smallest(numbers)) + " is the smallest")
    print(str(divisible(numbers)) + " numbers are divislbe by 3")
    return

def largest(numbers):
    """
    Input: A list of numbers
    Returns: Largest number in the list
    """
    largest = numbers[0]
    for number in numbers:
        if number > largest:
            largest = number
    return largest

def smallest(numbers):
    """
    Input: A list of numbers
    Returns: Smallest number in the list
    """
    smallest = numbers[0]
    for number in numbers:
        if number < smallest:
            smallest = number
    return smallest

def divisible(numbers):
    """
    Input: A list of numbers
    Returns: Total number of numbers divisible by 3
    """
    count = 0
    for number in numbers:
        if number % 3 == 0:
            count += 1
    return count

if __name__ == "__main__":
    numbers = []
    while True:
        number = input("Enter a number for the number list or press q to quit: ")
        if number == "q":
            break
        else:
            numbers.append(int(number))
    if numbers == []:
        print("No numbers entered. Quitting.")
    else:
        print("Number list entered: ", numbers)
        main(numbers)
