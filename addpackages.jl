const PACKAGES_TO_ADD = [
    "LinearAlgebra",
    "Plots",
    "Statistics",
    "Flux",
    "Zygote",
    "ParameterSchedulers",
    "Reexport",
    "Revise",
    "Functors",
    "Random"
]

# --- 2. Define the Project Directory ---
# @__DIR__ gives the directory of the current script.
const PROJECT_DIR = @__DIR__

println("--- Starting automatic package addition for project in: $(PROJECT_DIR) ---")

try
    using Pkg # Ensure Pkg module is loaded
    
    # Activate the environment. This will use or create a Project.toml in PROJECT_DIR.
    # It's important to activate the correct environment before adding packages.
    Pkg.activate(PROJECT_DIR)
    println("Project environment activated: $(Pkg.project().name).")

    # Add (and install) the specified packages.
    # Pkg.add() will resolve dependencies, download them, and update
    # Project.toml and Manifest.toml in the active environment.
    println("\nAdding/installing packages: $(join(PACKAGES_TO_ADD, ", "))...")
    Pkg.add(PACKAGES_TO_ADD)
    println("All specified packages added successfully!")
    
    println("\n--- Setup complete! ---")
    println("To use, navigate to $(PROJECT_DIR) in your terminal and run:")
    println("  julia --project=.")
    println("\nThen, in the Julia REPL, you can 'using YourPackage' or 'using $(PACKAGES_TO_ADD[1])'.")

catch e
    println(stderr, "An error occurred during package setup: ", e)
    println(stderr, "Please ensure you have an active internet connection and correct package names.")
    rethrow(e) # Re-throw the error to stop script execution
end