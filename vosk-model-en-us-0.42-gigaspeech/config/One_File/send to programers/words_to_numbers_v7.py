#MIT License

#Copyright (c) 2025 IncredibleGeek

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



import json
import re

# Load a JSON file (generic function for any JSON mappings)
def load_json_map(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found!")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}!")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error in {file_path}! {e}")

# Define mappings for small numbers (0-999)
SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000
}

# Convert a phrase representing a number less than 1000 to its numerical value
def words_to_number_less_than_thousand(phrase):
    if not phrase:
        return 0

    words = phrase.replace("-", " ").split()
    if not words:
        return 0

    total = 0
    current = 0

    for word in words:
        if word in SMALL_NUMBERS:
            value = SMALL_NUMBERS[word]
            if value == 100:
                current *= 100
            elif value >= 1000:
                current *= value
                total += current
                current = 0
            else:
                current += value
        else:
            return None  # Return None instead of raising an exception

    return total + current

# Main function to convert a word phrase to a number
def convert_numbers(phrase, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    # Load large_numbers_map if not provided
    if large_numbers_map is None:
        large_numbers_map = load_json_map("large_numbers_map.json")

    # Ignore fraction_map, symbs_map, and google_numbers for now
    # These can be used in future implementations if needed

    # Practical upper limit to prevent overflow
    PRACTICAL_LIMIT = 10**303  # Up to centillion (10^303)
    
    # Clean the input phrase
    if not isinstance(phrase, str):
        return None
    
    phrase = phrase.lower().strip()
    if not phrase:
        return None

    phrase = re.sub(r'\s+', ' ', phrase)  # Normalize spaces
    phrase = phrase.replace(" and ", " ")  # Remove "and"
    phrase = phrase.replace(",", "")  # Remove commas

    # Handle negative numbers
    is_negative = False
    if phrase.startswith("negative"):
        is_negative = True
        phrase = phrase[len("negative"):].strip()
        if not phrase:
            return None

    # Split the phrase into sections based on large number scales
    large_scales = sorted(
        large_numbers_map.items(),
        key=lambda x: int(x[1]),
        reverse=True
    )

    # Create a regex pattern to split on large number words
    large_scale_words = "|".join(re.escape(scale) for scale, _ in large_scales)
    pattern = f"\\b({large_scale_words})\\b"
    sections = re.split(pattern, phrase)

    total = 0
    current_section_value = 0
    current_scale = 1  # Default scale for numbers less than the smallest large scale

    # Process each section
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section in large_numbers_map:
            # This section is a large scale (e.g., "billion")
            scale_value = int(large_numbers_map[section])
            if current_section_value == 0:
                current_section_value = 1  # e.g., "billion" alone means "one billion"
            total += current_section_value * scale_value
            current_section_value = 0
            current_scale = 1  # Reset for the next section
        else:
            # This section is a number phrase (e.g., "one hundred and twenty-three")
            section_value = words_to_number_less_than_thousand(section)
            if section_value is None:
                return None
            current_section_value += section_value

    # Add any remaining value (e.g., numbers less than the smallest large scale)
    total += current_section_value * current_scale

    # Check if the result exceeds the practical limit
    if total > PRACTICAL_LIMIT:
        return None

    return -total if is_negative else total

# Helper function to parse a sequence of number words from a list
def parse_number_sequence(words, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    """Parse a sequence of number words into a single number, returning the number and the number of words consumed."""
    if not words:
        return None, 0

    # Join words into a phrase and try to parse it
    phrase = " ".join(words)
    number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
    if number is not None:
        return str(number), len(words)

    # Try parsing smaller sequences until we find a valid number
    for i in range(len(words), 0, -1):
        sub_phrase = " ".join(words[:i])
        number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
        if number is not None:
            return str(number), i
    return None, 0

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Load the large numbers map
    large_numbers_map = load_json_map("large_numbers_map.json")

    # Test cases
    test_phrases = [
        "one hundred and twenty-three",
        "nine hundred and ninety-nine thousand",
        "one billion",
        "one trillion two hundred and thirty-four billion five hundred and sixty-seven million",
        "negative one hundred",
        "one vigintillion",
        "billion",  # Edge case: just the scale word
        "hello"  # Should return None
    ]

    for phrase in test_phrases:
        result = convert_numbers(phrase, None, None, None, large_numbers_map)
        print(f"{phrase} -> {result}")