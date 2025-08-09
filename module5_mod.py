class NumberProcessor:
    def __init__(self):
        self.numbers = []

    def read_numbers(self, n):
        """Reads N numbers from the user"""
        for i in range(n):
            number = int(input(f"Enter number {i + 1}: "))
            self.numbers.append(number)

    def find_number(self, x):
        """Finds the index of X in the list of numbers. Returns -1 if not found."""
        if x in self.numbers:
            return self.numbers.index(x) + 1  # Return index (1-based)
        else:
            return -1
