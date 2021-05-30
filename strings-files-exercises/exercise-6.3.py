#In the text below, count how often the word "wood" occurs (using program code,
#of course). Capitals and lower case letters may both be used, and you have to
#consider that the word "wood" should be a separate word, and not part of
#another word. Hint: If you did the exercises from this chapter, you already
#developed a function that "cleans" a text. Combining that function with the
#`split()` function more or less solves the problem for you.

text = """How much wood would a woodchuck chuck
If a woodchuck could chuck wood?
He would chuck, he would, as much as he could,
And chuck as much as a woodchuck would
If a Mr. Smith could chuck wood\n\r\t."""

i = 0
cleaned = ''
while i < len(text):
    if not ((text[i] >= 'a' and text[i] <='z') or (text[i] >= 'A' and text[i] <= 'Z')):
        cleaned += ' '
    else:
        cleaned += text[i]
    i += 1

text_list = cleaned.lower().split(' ')
count = 0
for word in text_list:
    if word == 'wood':
        count += 1
print("The count is: ", count)