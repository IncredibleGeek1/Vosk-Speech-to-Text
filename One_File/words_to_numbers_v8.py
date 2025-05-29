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


import re
import json

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

# Define mappings for small numbers
SMALL_NUMBERS = {
    "negative": "-", "minus": "-",
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000
}

def format_number(number_str):
    """Format a number string with commas (e.g., '1247' → '1,247')."""
    try:
        number = int(number_str)
        return f"{number:,}"
    except ValueError:
        return number_str

def words_to_number_less_than_thousand(phrase):
    """Convert a phrase of number words to a number, handling digit-by-digit sequences."""
    if not phrase:
        return 0

    words = phrase.replace("-", " ").split()
    if not words:
        return 0

    # Check if all words are single digits (zero to nine)
    digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    if all(word in digit_words for word in words):
        # Concatenate digits
        digit_str = "".join(str(SMALL_NUMBERS[word]) for word in words)
        return int(digit_str) if digit_str else 0

    # Standard parsing for numbers < 1000
    total = 0
    current = 0
    for word in words:
        if word in SMALL_NUMBERS:
            value = SMALL_NUMBERS[word]
            if value == "-":
                return None
            if value == 100:
                current *= 100
            elif value >= 1000:
                current *= value
                total += current
                current = 0
            else:
                current += value
        else:
            return None
    return total + current

def convert_numbers(phrase, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    """Convert a number phrase to a number, handling digit sequences and standard phrases."""
    if large_numbers_map is None:
        large_numbers_map = load_json_map("large_numbers_map.json")

    PRACTICAL_LIMIT = 999999999

    if not isinstance(phrase, str):
        return None
    
    phrase = phrase.lower().strip()
    if not phrase:
        return None

    phrase = re.sub(r'\s+', ' ', phrase).replace(" and ", " ").replace(",", "")

    is_negative = False
    if phrase.startswith("negative") or phrase.startswith("minus"):
        is_negative = True
        phrase = phrase.lstrip("negative").lstrip("minus").strip()
        if not phrase:
            return None

    # Check if phrase is a digit-by-digit sequence
    words = phrase.split()
    digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    if all(word in digit_words for word in words):
        number = words_to_number_less_than_thousand(phrase)
        if number > PRACTICAL_LIMIT:
            return None
        return -number if is_negative else number

    large_scales = sorted(
        large_numbers_map.items(),
        key=lambda x: int(x[1]),
        reverse=True
    )

    large_scale_words = "|".join(re.escape(scale) for scale, _ in large_scales)
    pattern = f"\\b({large_scale_words})\\b"
    sections = re.split(pattern, phrase)

    total = 0
    current_section_value = 0
    current_scale = 1

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if section in large_numbers_map:
            scale_value = int(large_numbers_map[section])
            if current_section_value == 0:
                current_section_value = 1
            total += current_section_value * scale_value
            current_section_value = 0
            current_scale = 1
        else:
            section_value = words_to_number_less_than_thousand(section)
            if section_value is None:
                return None
            current_section_value += section_value

    total += current_section_value * current_scale

    if total > PRACTICAL_LIMIT or total < -PRACTICAL_LIMIT:
        return None

    return -total if is_negative else total

def parse_number_sequence(words, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    """Parse a sequence of number words into a formatted number string."""
    if not words:
        return None, 0

    is_negative = words[0].lower() in ["negative", "minus"]
    start_idx = 1 if is_negative else 0
    phrase = " ".join(words[start_idx:]).replace("-", " ")

    number = convert_numbers(phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
    if number is not None:
        return format_number(str(number)), len(words)

    for i in range(len(words) - start_idx, 0, -1):
        sub_phrase = " ".join(words[start_idx:start_idx + i]).replace("-", " ")
        number = convert_numbers(sub_phrase, fraction_map, symbs_map, google_numbers, large_numbers_map)
        if number is not None:
            return format_number(str(number)), start_idx + i
    return None, 0

def convert_numbers_chained(text, fraction_map=None, symbs_map=None, google_numbers=None, large_numbers_map=None):
    """Process chained input, converting number word sequences to formatted numbers."""
    text = re.sub(r'\s+', ' ', text.lower().strip()).replace(" and ", " ").replace("-", " ")
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        if words[i] in SMALL_NUMBERS or words[i] in large_numbers_map:
            number_words = []
            while i < len(words) and (words[i] in SMALL_NUMBERS or words[i] in large_numbers_map):
                number_words.append(words[i])
                i += 1
            number_str, words_consumed = parse_number_sequence(number_words, fraction_map, symbs_map, google_numbers, large_numbers_map)
            if number_str:
                result.append(number_str)
            else:
                result.extend(number_words)
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)

# Example usage
if __name__ == "__main__":
    large_numbers_map = {"million": 1000000, "billion": 1000000000}
    test_cases = [
        "one two four seven two four seven",
        "three million four hundred forty-five thousand three hundred twenty-four",
        "three million hello one two three world six",
        "negative one hundred",
        "million"
    ]
    for phrase in test_cases:
        result = convert_numbers_chained(phrase, None, None, None, large_numbers_map)
        print(f"{phrase} -> {result}")