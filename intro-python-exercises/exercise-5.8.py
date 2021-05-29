# A, B, C, and D are all different digits. The number DCBA is equal to 4 times
# the number ABCD. What are the digits? Note: to make ABCD and DCBA conventional
# numbers, neither A nor D can be zero. Use a quadruple-nested loop.

# Solve 4*ABCD == DCBA
def nested_nest():
    
    """
    A = 2
    B = 1
    C = 7
    D = 8
    """
    
    for A in range(1,10):
        for B in range(1,10):
            for C in range(1,10):
                for D in range(1,10):
                    if 4*((10**3)*A + (10**2)*B + (10)*C + D) == ((10**3)*D + (10**2)*C + (10)*B + A):
                        print(f"A = {A} \n B = {B} \n C = {C} \n D = {D}")
                        return 
    return "No such number"

nested_nest()