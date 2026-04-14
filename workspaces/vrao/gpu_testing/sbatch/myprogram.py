#!/usr/bin/env python3
#
# Print the sum of integers from 1 to whatever is given on the command line
# or to 100 if nothing is given on the command line

import sys

max = 100  # Default max if no arguments given
if len(sys.argv) == 2:
    max = int(sys.argv[1])  # We were given an argument, use that instead!

sum = 0  # Running total
for i in range(1, max + 1, 1):
    sum += i
print(f"The sum of integers from 1 to {max} is {sum}")
