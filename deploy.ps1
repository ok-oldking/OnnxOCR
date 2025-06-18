# Define variables for ease of use
$DistFolder = "dist"
$BuildFolder = "build"

# Delete dist folder if it exists
if (Test-Path $DistFolder) {
    Write-Host "Deleting $DistFolder folder..."
    Remove-Item -Path $DistFolder -Recurse -Force
} else {
    Write-Host "$DistFolder folder not found."
}

# Delete build folder if it exists
if (Test-Path $BuildFolder) {
    Write-Host "Deleting $BuildFolder folder..."
    Remove-Item -Path $BuildFolder -Recurse -Force
} else {
    Write-Host "$BuildFolder folder not found."
}

# Build the project
Write-Host "Building the project..."
try {
    python -m build
    Write-Host "Build successful."
} catch {
    Write-Host "Error during build:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    exit 1 # Exit with an error code
}

# Upload to PyPI (or Test PyPI)
Write-Host "Uploading to PyPI..."
try {
    # Use the command for uploading to official PyPI
    python -m twine upload dist/*

    # OR use the command for uploading to Test PyPI for testing
    # python -m twine upload --repository testpypi dist/*

    Write-Host "Upload successful."
} catch {
    Write-Host "Error during upload:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    exit 1 # Exit with an error code
}
