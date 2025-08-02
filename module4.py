
# module4.py

N = int(input("Enter a positive integer N: "))

numbers = []
for i in range(N):
    num = int(input(f"Enter number {i + 1}: "))
    numbers.append(num)

X = int(input("Enter a number to search (X): "))

if X in numbers:
    print(numbers.index(X) + 1)  # Output index starting from 1
else:
    print(-1)
