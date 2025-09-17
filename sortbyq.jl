using CSV, DataFrames

# Check if filename argument is provided
if length(ARGS) == 0
    println("Usage: julia sort_csv.jl <filename>")
    exit(1)
end

# Get filename from command line argument
input_filename = ARGS[1]

# Check if file exists
if !isfile(input_filename)
    println("Error: File '$input_filename' not found")
    exit(1)
end

# Read the CSV file
df = CSV.read(input_filename, DataFrame)

# Sort by q column first, then by p column
df_sorted = sort(df, [:q, :p])

# Generate output filename
base_name = splitext(input_filename)[1]
output_filename = base_name * "_sorted.csv"

# Write the sorted data to a new CSV file
CSV.write(output_filename, df_sorted)

println("File sorted by q column and saved as '$output_filename'")

# Optional: Display the first few rows to verify
# println("\nFirst 10 rows of sorted data:")
# println(first(df_sorted, 10))
