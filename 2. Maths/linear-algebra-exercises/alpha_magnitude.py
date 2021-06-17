def alpha_magnitude(alpha, vec):
    
    res = alpha*np.array(vec)
    magnitude = np.linalg.norm(res)
    direction = res/magnitude

    return (f"The magnitude of ({alpha}*({vec[0]}i + {vec[1]}j) + " 
                                f"is {magnitude} and the direction is " 
                                f"{direction[0]}i + {direction[1]}j")
    #returns the resulting magnitude, if the direction has changed and what has happened to the vector
vector = [[3],[4]]
print(alpha_magnitude(1, vector))       
print(alpha_magnitude(2, vector))       # same vector direction
print(alpha_magnitude(-1, vector))      # opposite vector direction
print(alpha_magnitude(0.5, vector))     # |alpha*vec| decreases   