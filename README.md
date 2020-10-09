# Mandelbrot-Julia-Generator
A simple program using OpenCL to generate images of Mandelbrot and Julia sets in Linux or macOS using ImageWriter library

# Instruction
- Update makefile in app folder according to the system (Linux or macOS)
- Change params in params.txt
  - First line: Height and width
  - Second line: Maximum iterations
  - Third line: Max length squared
  - Fourth line: Real part of the minimum, real part of the maximum
  - Fifth line: Imaginary part of the minimum, imaginary part of the maximum
  - Sixth line: Julia point
  - Seventh line: First color in RGB
  - Eighth line: Second color in RGB
  - Ninth line: Third color in RGB
  Example:
  2000 2000
  200
  4
  -0.5 0.5
  -0.5 0.5
  -0.4 0.6
  1 1 0.7
  0 0 0.3
  1 0.7 0
- Make and run
  Example: 
    - For Mandelbrot: MandelbrotJuliaGenerator M params.txt imageOut.png
    - For Julia:      MandelbrotJuliaGenerator J params.txt imageOut.png
