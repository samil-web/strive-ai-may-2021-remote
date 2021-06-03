# Factorial
def factorial(number):                                
    #Your Code Here
    if number in [0,1]:
        return 1
    
    else:
        fact = 1
        while number > 1:
            fact *= number
            number -= 1
        return fact