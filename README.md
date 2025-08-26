# XRD Peak Analysis Engine

A Python-based tool for analyzing **X-ray diffraction (XRD) data**, including background subtraction, peak detection, lattice parameter estimation, composition analysis, and grain size calculation.

This engine processes raw XRD data files, extracts relevant structural information, and visualizes the analysis results with annotated diffraction patterns.

## Features

### Data Handling
- **Load Experimental Data**: Reads XRD data files, extracts metadata (wavelength), and loads 2θ vs intensity counts.  
- **Background Modeling**: Fits a quadratic model to background radiation using non-peak regions.  

### Peak Analysis
- **Initial Peak Detection**: Conservative thresholds using `scipy.signal.find_peaks`.  
- **Detailed Peak Characterization**: Calculates  
  - Peak positions (2θ) and errors  
  - FWHM (full width at half maximum)  
  - Peak area & area error  
  - Interplanar spacing *d* (with error)  
- **Gaussian Fitting**: Fits Gaussian curves to peaks for refined analysis.  

### Structural Properties
- **Lattice Parameter Estimation**: Uses selected Miller indices (hkl) to estimate lattice constants.  
- **Composition Calculation**: Determines alloy composition (Cu-Au system) based on Vegard’s law.  
- **Grain Size Estimation**: Applies Scherrer’s equation with uncertainty propagation.  

### Visualization
- Annotated diffraction plots including:  
  - Raw data, background, and background-subtracted curves  
  - Fitted Gaussian peaks  
  - Labels with FWHM & area  
  - Composition and grain size displayed on the figure  
  - Log-scaled y-axis for enhanced visibility of weaker peaks  

## Output

### Printed Results (Dictionary)
- Peak positions, d-spacings, FWHM, areas (with errors)  
- Average lattice parameter composition (with error)  
- Average grain size (with error)  

### Plot
- Raw vs background vs net counts  
- Annotated peaks with Gaussian fits  
- Composition and grain size summary  

## Dependencies

- `numpy`  
- `matplotlib`  
- `scipy`  
