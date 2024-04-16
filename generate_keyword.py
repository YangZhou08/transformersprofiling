import random
import string

def generate_random_string(length):
    # Define the possible characters in the string: uppercase, lowercase, and digits
    characters = string.ascii_letters + string.digits
    # Randomly choose characters from the characters string for the specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

# Example usage
# length_of_string = 10  # Specify the desired length of the string 
length_of_string = 21 
past_used_file = open("requeueneededtxt.txt", "r") 
past_used_string = past_used_file.readlines() 
metcondition = False 
random_string = None 

while metcondition == False: 
    random_string = generate_random_string(length_of_string) 
    if random_string not in past_used_string: 
        metcondition = True 
        past_used_file.close() 

past_used_file = open("requeueneededtxt.txt", "a") 
past_used_file.write(random_string + "\n") 

past_used_file.close() 
print("Randomx string {}".format(random_string)) 
