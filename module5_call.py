from module5_mod import NumberProcessor

def main():
    # Ask for the number of inputs (N)
    n = int(input("Enter the number of elements (N): "))

    # Create an instance of NumberProcessor
    processor = NumberProcessor()

    # Read N numbers
    processor.read_numbers(n)

    # Ask for the number to search (X)
    x = int(input("Enter the number to search for (X): "))

    # Find the index of X
    result = processor.find_number(x)

    # Output the result
    if result == -1:
        print("-1")  # If the number is not found
    else:
        print(f"The number {x} is at index {result}.")

if __name__ == "__main__":
    main()
