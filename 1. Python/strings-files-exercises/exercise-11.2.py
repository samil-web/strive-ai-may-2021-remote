def clean_text(text):
    cleaned = ''
    for char in text.lower():
        if not ((char >= 'a' and char <= 'z') or char == "'"):
            cleaned += ' '
        else:
            cleaned += char
    
    return cleaned
words = {}
fp = open( "text-files/blakepoems.txt")
for line in fp.readlines():
    cleaned = clean_text(line)
    for word in cleaned.split():
        words[word] = words.get(word, 0) + 1


fp.close()
print(words)