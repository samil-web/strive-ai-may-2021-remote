
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
            seek += 1
        return primes

if __name__ == "__main__":
    number = int(input("Enter number below which list of primes is to printed for: "))
    print(primes_list(number))