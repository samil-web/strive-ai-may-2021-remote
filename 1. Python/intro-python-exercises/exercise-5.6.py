# A prime number is a positive integer that is dividable by exactly two
# different numbers, namely 1 and itself. The lowest (and only even) prime
# number is 2. The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23,
# and 29. Write a function that returns a list off all prime numbers below a
# given number.

# Hint: In a loop where you test the possible dividers of the number, you can
# conclude that the number is not prime as soon as you encounter a number other
# than 1 or the number itself that divides it. However, you can only conclude
# that it actually is prime after you have tested all possible dividers.

# What is the challenge here? You have to try to optimize your code and try to
# make it work for the highest prime number you can encounter before you run out
# of memory. For low numbers you should know how to do it already

import math

def primes_list(number):
    if number < 2:
        return []

    elif number == 2:
        return [2]
    
    else:
        seek = 3
        primes = [2]
        while seek < number:
            
            # Check if divisible from current list of primes
            for divisor in primes: 
                if seek % divisor == 0:
                    break

            # Check if for loop fully executed
            if (divisor == primes[-1]) and (seek % divisor != 0): 
                primes.append(seek)
            seek += 2
        return primes

if __name__ == "__main__":
    number = int(input("Enter number below which list of primes is to printed for: "))
    print(primes_list(number))