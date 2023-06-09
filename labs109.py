def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust

def is_ascending(items):
    # base case
    if len(items) == 0 or len(items) == 1:
        return True

    for i in range(1, len(items)):
        if items[i] <= items[i - 1]:
            return False
    return True

def riffle(items, out = True):
    mid = int(len(items)/2)
    a = []
    b = []
    if out:
        a = items[:mid]
        b = items[mid:]
    else:
        b = items[:mid]
        a = items[mid:]
    i=0
    j=0
    for k in range(mid):
        items[i] = a[j]
        i = i+1
        items[i] = b[j]
        i = i+1
        j = j+1
    return items

def only_odd_digits(n):
    string_number = str(n)

    for i in string_number:
        i = int(i)

        if i % 2 == 0:
            return False
    return True


def is_cyclops(n):
    s = str(n)
    if len(s) % 2 == 1 and s[len(s) // 2] == '0':
        for i in range(len(s)):
            if i != len(s) // 2 and s[i] == '0':
                return False
        return True
    return False


def domino_cycle(tiles):
    if tiles == None : return False
    if tiles==[]:return True
    starting_num = tiles[0][0]
    ending_num = tiles[-1][1]
    if starting_num != ending_num: return False

    for i in range(len(tiles) - 1):
        if tiles[i][1] != tiles[i + 1][0]: return False
    return True

def colour_trio(colours):
    while len(colours) > 1:
        row = ""
        for i in range(len(colours)-1):
            c1, c2, = colours[i], colours[i+1]
            if c1 == c2:
                row += c1
            else:
                row += "ybr".replace(c1,"").replace(c2,"")
        else: colours = row
    return colours

def taxi_zum_zum(moves):
    dirs = ["North", "East", "South", "West"]
    dx_dy = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    x, y = 0, 0
    curr = 0
    for i in moves:
        if i == 'R':
            curr = (curr + 1) % 4
        elif i == 'L':
            curr = (curr - 1) % 4
        else:
            x += dx_dy[curr][0]
            y += dx_dy[curr][1]
    return (x, y)

def extract_increasing(digits):
    result = []
    current = 0
    previous = -1
    for i in range(len(digits)):
        d = int(digits[i])
        current = 10 * current + d
        if current > previous:
            result.append(current)
            previous = current
            current = 0
    return result

def give_change(amount, coins):
    List = []
    i = 0
    while(amount > 0):
        if amount >= coins[i]:
            List.append(coins[i])
            amount = amount - coins[i]
        else:
            i=i+1
    return List

def safe_squares_rooks(n, rooks):
    safe_cols = set(x for x in range(0, n))
    safe_rows = set(x for x in range(0, n))
    for x, y in rooks:
        safe_cols.discard(x)
        safe_rows.discard(y)
        if len(safe_rows) == 0 or len(safe_cols) == 0:
            return 0
    return len(safe_cols) * len(safe_rows)

def knight_jump(knight, start, end):
    knight = list(knight)
    knight.sort()
    arr = []
    for i in range(len(start)):
        arr.append(abs(start[i] - end[i]))
        arr.sort()
    if arr == knight:
        return True
    else:
        return False

def seven_zero(n):
    d = 1
    ans = 0
    while True:
        if n%2 == 0 or n%5 == 0:
            k = 1
            while k <= d:
                val = int(k * '7' + (d-k) * '0')
                if val%n == 0:
                    ans = val
                    break
                k += 1
        else:
            val = int(d * '7')
            ans = val if val%n == 0 else 0
        d += 1
        if ans > 0:
            return ans

def can_balance(items):
    n = len(items)
    for i in range(n):
        left_torque = 0
        right_torque = 0
        for j in range(i):
            left_torque += items[j]*(i-j)

        for j in range(i+1, n):
            right_torque += items[j]*(j-i)

        if left_torque == right_torque:
            return i

    return -1

def group_and_skip(n, out, ins):
    disc_val = []
    while n != 0:
        disc_val.append(n % out)
        n = n // out
        n = n*ins
    return disc_val

def count_growlers(animals):
    dogs = []
    cats = []
    d = 0
    c = 0
    for animal in animals:
        if(animal == 'dog' or animal == 'god'):
            d += 1
            dogs.append(d)
            cats.append(c)
        else:
            c += 1
            cats.append(c)
            dogs.append(d)
    ans = 0
    i = 0
    length = len(animals)
    while i < len(animals):
        if animals[i] == 'dog' or animals[i] == 'cat':
            if i > 0 and dogs[i-1] > cats[i-1]:
                ans += 1
        elif animals[i] == 'god' or animals[i] == 'tac':
            if (dogs[length-1] - dogs[i]) > (cats[length-1] - cats[i]):
                ans += 1
        i += 1
    return ans

def bulgarian_solitaire(piles, k):
    def CheckIfNotZero(number):
        if number != 0:
            return (True)
        return (False)

    def check_for_first_k_numbers(piles, k):
        n = len(piles)
        countOccurances = {}
        for pile in piles:
            countOccurances[pile] = 0
        if len(countOccurances.keys()) == n and n == k:
            return (True)
        return (False)

    if sum(piles) != k * (k + 1) // 2:
        return False
    else:
        count = 0
        while not check_for_first_k_numbers(piles, k):
            for i in range(len(piles)):
                piles[i] -= 1
            mod_piles = len(piles)
            piles = list(filter(CheckIfNotZero, piles))
            piles.append(mod_piles)
            count = count + 1
    return(count)


def scylla_or_charybdis(moves, n):
    min_step = float('inf')
    k = -1
    for i in range(1, len(moves) + 1):
        nemesis_pos = 0
        curr_step = 0
        for j in range(i - 1, len(moves), i):
            if moves[j] == '+':
                nemesis_pos += 1
            else:
                nemesis_pos -= 1

            curr_step += 1
            if nemesis_pos == n or nemesis_pos == -(n):
                if curr_step < min_step:
                    min_step = curr_step
                    k = i
    return k


def tukeys_ninthers(items):
    if len(items) == 1:
        return items[0]
    triplets = [items[i:i + 3] for i in range(0, len(items), 3)]
    medians = [sorted(triplet)[1] for triplet in triplets]
    return tukeys_ninthers(medians)


def verify_betweenness(perm, constraints):
    n = len(perm)
    inv = [0] * n
    for i in range(n):
        inv[perm[i]] = i
    for constraint in constraints:
        a, b, c = constraint
        if not (inv[a] < inv[b] < inv[c] or inv[c] < inv[b] < inv[a]):
            return False
    return True


from collections import defaultdict
from itertools import combinations

def count_troikas(items):
    troika_count = 0
    positions = defaultdict(list)

    for idx, item in enumerate(items):
        positions[item].append(idx)

    for item, pos_list in positions.items():
        for i, j in combinations(pos_list, 2):
            k = j + (j - i)
            if k < len(items) and items[k] == items[i]:
                troika_count += 1
    return troika_count

def three_summers(items, goal):
    def two_summers(items, goal):
        left = 0
        right = len(items) - 1

        while left < right:
            current_sum = items[left] + items[right]

            if current_sum == goal:
                return True
            elif current_sum < goal:
                left += 1
            else:
                right -= 1

        return False

    for i in range(len(items) - 2):
        remaining_sum = goal - items[i]
        if two_summers(items[i + 1:], remaining_sum):
            return True

    return False

def sum_of_two_squares(n):
    left = 1
    right = int(n ** 0.5)

    while left <= right:
        current_sum = left ** 2 + right ** 2

        if current_sum == n:
            return (right, left) if right >= left else (left, right)
        elif current_sum < n:
            left += 1
        else:
            right -= 1

    return None

def count_carries(a, b):
    carry = 0
    carry_count = 0

    while a > 0 or b > 0:
        digit_sum = a % 10 + b % 10 + carry
        carry = digit_sum // 10

        if carry > 0:
            carry_count += 1

        a //= 10
        b //= 10
    return carry_count

def candy_share(candies):
    def stop(l):
        for i in l:
            if i >= 2:
                return False
        return True

    n = len(candies)
    c = 0
    while True:
        if stop(candies):
            return c
        c += 1
        new_candies = [0] * n
        for i in range(n):
            if candies[i] >= 2:
                candies[i] -= 2
                if i == n - 1:
                    new_candies[i - 1] += 1
                    new_candies[0] += 1
                else:
                    new_candies[i - 1] += 1
                    new_candies[i + 1] += 1
            new_candies[i] += candies[i]
        candies = new_candies


def squares_intersect(s1, s2):
    sq1x1 = s1[0]
    sq1y1 = s1[1]
    sq1x2 = sq1x1 + s1[2]
    sq1y2 = sq1y1 + s1[2]

    sq2x1 = s2[0]
    sq2y1 = s2[1]
    sq2x2 = sq2x1 + s2[2]
    sq2y2 = sq2y1 + s2[2]

    if sq1x2 < sq2x1 or sq2x2 < sq1x1:
        return False

    if sq1y2 < sq2y1 or sq2y2 < sq1y1:
        return False

    return True

def nearest_smaller(items):
    result = []
    curr_ele_index = 0

    while len(result) < len(items):
        smallest_ele_left = 'x'
        smallest_ele_right = 'x'
        left_index = curr_ele_index - 1

        while left_index >= 0:
            if items[left_index] < items[curr_ele_index]:
                smallest_ele_left = left_index
                break
            left_index -= 1

        right_index = curr_ele_index + 1

        while right_index < len(items):
            if items[right_index] < items[curr_ele_index]:
                smallest_ele_right = right_index
                break
            right_index += 1

        if smallest_ele_left == 'x':
            if smallest_ele_right == 'x':
                result.append(items[curr_ele_index])
            else:
                result.append(items[smallest_ele_right])
        else:
            if smallest_ele_right == 'x':
                result.append(items[smallest_ele_left])
            else:
                dist_left = curr_ele_index - smallest_ele_left
                dist_right = smallest_ele_right - curr_ele_index

                if dist_left < dist_right:
                    result.append(items[smallest_ele_left])
                elif dist_right < dist_left:
                    result.append(items[smallest_ele_right])
                else:
                    smaller_ele = min(items[smallest_ele_left], items[smallest_ele_right])
                    result.append(smaller_ele)

        curr_ele_index += 1

    return result

def remove_after_kth(items, k=1):
    count = {}
    result = []

    for item in items:
        count[item] = count.get(item, 0) + 1

        if count[item] <= k:
            result.append(item)

    return result


def brussels_choice_step(n, mink, maxk):
    n_str = str(n)
    results = []

    for i in range(len(n_str)):
        for j in range(i + mink, min(i + maxk + 1, len(n_str) + 1)):
            substr = n_str[i:j]
            substr_int = int(substr)

            if substr_int % 2 == 0:
                results.append(int(n_str[:i] + str(substr_int // 2) + n_str[j:]))

            results.append(int(n_str[:i] + str(2 * substr_int) + n_str[j:]))

    return sorted(results)

def first_preceded_by_smaller(items, k=1):
    for i in range(len(items)):
        count = 0
        for j in range(i):
            if items[j] < items[i]:
                count += 1
        if count >= k:
            return items[i]
    return None

def eliminate_neighbours(items):
    if len(items)==1:
        return 1
    items = list(items)
    n = len(items)
    counter = 0
    for i in range(1, n + 1):
        if i in items:
            counter += 1
            if len(items)==1:
                items.pop(0)
                break
            index1 = items.index(i)
            index2 = index1 - 1
            if index2 < 0 or ((index1 + 1) < len(items) and items[index1 + 1] > items[index2]):
                index2 = index1 + 1
            value = items[index2]
            if index1 > index2:
                index1 = index2
            items.pop(index1)
            items.pop(index1)
            if value == n:
                break
    return counter

def count_and_say(digits):
    result = ""
    count = 0
    current_digit = None

    for digit in digits:
        if digit == current_digit:
            count += 1
        else:
            if current_digit is not None:
                result += str(count) + current_digit
            current_digit = digit
            count = 1

    result += str(count) + current_digit
    return result

def safe_squares_bishops(n, bishops):
    covered_squares = set()

    for bishop in bishops:
        r, c = bishop
        for i in range(n):
            for j in range(n):
                if abs(r - i) == abs(c - j):
                    covered_squares.add((i, j))

    total_squares = n * n
    safe_squares = total_squares - len(covered_squares)
    return safe_squares

def counting_series(n):
    block_size = 9
    digits = 1

    while n >= block_size * digits:
        n -= block_size * digits
        block_size *= 10
        digits += 1

    num = 10 ** (digits - 1) + n // digits

    digit = str(num)[n % digits]
    return int(digit)

def reverse_vowels(text):
    vowels = []
    result = ""

    for char in text:
        if char.lower() in 'aeiou':
            vowels.append(char)

    for char in text:
        if char.lower() in 'aeiou':
            vowel = vowels.pop()
            if char.isupper():
                result += vowel.upper()
            else:
                result += vowel.lower()
        else:
            result += char

    return result

from fractions import Fraction
from collections import deque

def fractran(n, prog, giveup=1000):
    sequence = [n]
    counter = 0

    while counter < giveup:
        current_state = sequence[-1]
        halt = True

        for fraction in prog:
            numerator, denominator = fraction
            result = current_state * Fraction(numerator, denominator)

            if result.denominator == 1:
                sequence.append(result.numerator)
                halt = False
                break

        if halt:
            break

        counter += 1

    return sequence


def calkin_wilf(n):
    queue = deque()
    queue.append(Fraction(1, 1))

    for _ in range(n):
        fraction = queue.popleft()
        p, q = fraction.numerator, fraction.denominator

        left_fraction = Fraction(p, p + q)
        right_fraction = Fraction(p + q, q)

        queue.append(left_fraction)
        queue.append(right_fraction)

    return fraction

def nthRational(n):
    frac = [0, 1]
    if n > 0:
        nthRational(int(n / 2))

    frac[~n & 1] += frac[n & 1]

    return frac

def nearest_polygonal_number(n, s):
    def s_gonal_number(i):
        return ((s - 2) * i**2 - (s - 4) * i) // 2

    left = 1
    right = n

    while left <= right:
        mid = (left + right) // 2
        current_number = s_gonal_number(mid)
        next_number = s_gonal_number(mid + 1)

        if current_number <= n <= next_number or current_number >= n >= next_number:
            if abs(n - current_number) <= abs(n - next_number):
                return current_number
            else:
                return next_number
        elif current_number < n:
            left = mid + 1
        else:
            right = mid - 1

    return s_gonal_number(left)

def subtract_square(queries):
    max_query = max(queries)
    heat = [False] * (max_query + 1)

    for i in range(1, max_query + 1):
        for j in range(1, int(i ** 0.5) + 1):
            if not heat[i - j * j]:
                heat[i] = True
                break

    return [heat[query] for query in queries]

def reverse_ascending_sublists(items):
    reversed_list = []
    sublist = []

    for i in range(len(items)):
        sublist.append(items[i])

        if i == len(items) - 1 or items[i] >= items[i + 1]:
            reversed_list.extend(sublist[::-1])
            sublist = []

    return reversed_list


def count_divisibles_in_range(start, end, n):
    if n == 1:
        return end - start + 1
    count = 0
    if start % n == 0:
        count += 1
    count += end // n - start // n
    return count

def bridge_hand_shape(hand):
    suit_order = {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}
    counts = [0, 0, 0, 0]

    for rank, suit in hand:
        counts[suit_order[suit]] += 1

    return counts

def frequency_sort(items):
    frequency = {}
    for item in items:
        frequency[item] = frequency.get(item, 0) + 1

    sorted_items = sorted(items, key=lambda x: (-frequency[x], x))

    return sorted_items

def fibonacci_sum(n):
    fibonacci = [0, 1]

    while fibonacci[-1] <= n:
        fibonacci.append(fibonacci[-1] + fibonacci[-2])

    result = []
    i = len(fibonacci) - 1

    while n > 0 and i >= 0:
        if fibonacci[i] <= n and (not result or fibonacci[i] != result[-1] + fibonacci[i]):
            result.append(fibonacci[i])
            n -= fibonacci[i]
        i -= 1

    return sorted(result, reverse=True)

words = open("words_sorted.txt")

def words_with_letters(words, letters):
    def is_subsequence(subseq, word):
        i = 0
        j = 0

        while i < len(subseq) and j < len(word):
            if subseq[i] == word[j]:
                i += 1
            j += 1

        return i == len(subseq) and i != 0

    result = []

    for word in words:
        if is_subsequence(letters, word):
            result.append(word)

    return result

def words_with_given_shape(words, shape):
    def get_word_shape(word):
        shape = []
        for i in range(len(word) - 1):
            if word[i] < word[i + 1]:
                shape.append(1)
            elif word[i] == word[i + 1]:
                shape.append(0)
            else:
                shape.append(-1)
        return shape

    result = []

    for word in words:
        word_shape = get_word_shape(word)
        if word_shape == shape:
            result.append(word)

    return result

def substitution_words(pattern, words):
    matching_words = []

    for word in words:
        if len(word) != len(pattern):
            continue

        mapping = {}
        used_letters = set()

        for i in range(len(word)):
            pattern_char = pattern[i]
            word_char = word[i]

            if pattern_char in mapping:
                if mapping[pattern_char] != word_char:
                    break
            else:
                if word_char in used_letters:
                    break
                mapping[pattern_char] = word_char
                used_letters.add(word_char)
        else:
            matching_words.append(word)

    return matching_words

def unscramble(words, word):
    unscrambled_words = []
    sorted_word = sorted(word[1:-1])
    sorted_word_str = ''.join(sorted_word)

    for original_word in words:
        if len(original_word) == len(word) and original_word[0] == word[0] and original_word[-1] == word[-1]:
            sorted_original_word = sorted(original_word[1:-1])
            if sorted_word == sorted_original_word:
                unscrambled_words.append(original_word)

    return unscrambled_words

def postfix_evaluate(items):
    stack = []
    for item in items:
        if isinstance(item, int):
            stack.append(item)
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()

            if item == '+':
                result = operand1 + operand2
            elif item == '-':
                result = operand1 - operand2
            elif item == '*':
                result = operand1 * operand2
            elif item == '/':
                if operand2 != 0:
                    result = operand1 // operand2
                else:
                    result = 0

            stack.append(result)

    return stack[0]

def autocorrect_word(word, words, df):
    min_distance = float('inf')
    closest_word = None

    for w in words:
        if len(w) != len(word):
            continue

        distance = sum(df(a, b) for a, b in zip(word, w))

        if distance < min_distance:
            min_distance = distance
            closest_word = w

    return closest_word


































