import random

# This list contains the random responses
random_responses = ["That is so interesting, please tell me more.",
                   "I see, keep on!",
                   "Why do you say that?",
                   "Funny weather we've been having around, don't you think?",
                   "Let's get to know eachother!",
                   "What was the last movie you watched?"]

print("Hi, I'm Saba, the simple robot.")
print("You can end this conversation at any time by typing 'bye'")
print("After typing each answer, press 'enter'")
print("How are you today?")

while True:
    # wait for the user to enter some text
    user_input = input("> ")
    if user_input.lower() == "bye":
        # if they typed in 'bye' (or even BYE, ByE, byE etc.), break out of the loop
        break
    else:
        response = random.choices(random_responses)[0]
    print(response)

print("It was nice talking to you, goodbye my friend!")