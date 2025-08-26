# Importing all the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# Function to load the diffraction data from file and extract experimental parameters required
def load_xrd_data(file_path):
    """ Load XRD data from file and extract experimental parameters.

    Args:
    file_path: Path to the XRD data file.

    Returns:
    Wavelength, angles, and counts. """
    
    #Opens the file and reads all lines one by one
    with open(file_path, "r") as file:
        lines = file.readlines()
        wavelength = None
        # Finds the wavelength in the metadata
        for line in lines:
            if line.startswith("Wavelength"):
                # Converts the wavelength to meters from angstoms
                wavelength = float(line.split("=")[1].strip()) * 1e-10
                break
        # Raises an error if the wavelength cannot be found
        if wavelength is None:
            raise ValueError("Wavelength not found in file header")

    #Skips the header lines and loads the actual data
    skip_lines = lines.index(next(line for line in lines if "&END" in line)) + 2
    #Extracts the angles and counts from the file
    angles, counts = np.genfromtxt(file_path, skip_header=skip_lines, unpack=True)
    return wavelength, angles, counts

# A Function to detect the diffraction peaks from the data
def detect_peaks(counts, height=150, prominence=20, width=3):
    """ Initial peak detection using conservative thresholds.

    Args:
    counts: Counts data.
    height: Height threshold.
    prominence: Prominence threshold.
    width: Width threshold.

    Returns:
    Indices of detected peaks. """

    #Uses the find_peaks function from scipy to detect peaks
    peaks, _ = find_peaks(counts, height=height, prominence=prominence, width=width)
    return peaks

# Function which models the theta dependent background radiation
def background_model(two_theta, base, center):
    """Quadratic model for X-ray background radiation.

    Args:
    two_theta: Angles data.
    base: Base parameter.
    center: Center parameter.

    Returns:
    Background radiation values.
    """
    # Calculates the background radiation using the quadratic model
    return base - ((two_theta - center)**2)/100

#This function creates a mask excluding areas around detected peaks
# I used this "https://stackoverflow.com/questions/72310246/numpy-fastest-way-to-create-mask-array-around-indexes" for assistance but did not copy any code directly, was more used so i could understand the logic
def create_peak_mask(angles, peak_indices, window=3.0):

    """   Create mask excluding regions around detected peaks.

    Args:
    angles: Angles data.
    peak_indices: Indices of detected peaks.
    window: Window size.

    Returns:
    Mask values.
    """
    #Calculates the angle step
    angle_step = angles[1] - angles[0]
    #Intially creates the mask with all "True" values
    mask = np.ones_like(angles, dtype=bool)
    # Calculates the number of points to exclude around each peak
    exclude_points = int(window/(2*angle_step))
    #Sets the mask to "False" for the excluded points
    for idx in peak_indices:
        start = max(0, idx - exclude_points)
        end = min(len(angles), idx + exclude_points)
        mask[start:end] = False
    return mask

#This function fits the background model to non-peak regions
def fit_background(angles, counts, mask):
    """
    Fit background model to non-peak regions.

    Args:
    angles: Angles data.
    counts: Counts data.
    mask: Mask values.

    Returns:
    Background parameters and curve."""
    
    # Prepares the parameters for the background model
    p0 = [np.median(counts[mask]), np.median(angles[mask])]
    # Fits the background model using curve_fit
    popt, pcov = curve_fit(background_model, angles[mask], counts[mask], p0=p0)
    # Calculates the background curve
    bg_curve = background_model(angles, *popt)
    return popt, pcov, bg_curve

# A Function to calculate the Gaussian function
def gaussian(x, a, x0, sigma):
    """
    Gaussian function.

    Args:
    x: Input values.
    a: Amplitude.
    x0: Center.
    sigma: Standard deviation.

    Returns:
    Gaussian values.
    """
    # Calculates the Gaussian function
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# A Function to perform  peak analysis on the cleaned data
def analyze_peaks(angles, net_counts, height=20, prominence=10, width=2):
    """
    Perform detailed peak analysis on background-subtracted data.

    Args:
    angles: Angles data.
    net_counts: Net counts data.
    height: Height threshold.
    prominence: Prominence threshold.
    width: Width threshold.

    Returns:
    Peaks, FWHMs, areas, and widths.
    """
    #Detects peaks in the cleaned data
    peaks, _ = find_peaks(net_counts, height=height, prominence=prominence, width=width)
    # creates lists to store the results
    fwhms = []
    areas = []
    widths = []
    # Calculates the angle step
    angle_step = angles[1] - angles[0]
    # Checks if any peaks were detected
    if len(peaks) > 0:
        # Calculates the width of each peak
        width_data = peak_widths(net_counts, peaks, rel_height=0.5)
        fwhms = width_data[0] * angle_step
        # Calculates the area of each peak
        for peak_idx in peaks:
            height = net_counts[peak_idx]
            threshold = 0.05 * height
            left = peak_idx
            while left > 0 and net_counts[left] > threshold:
                left -= 1
            right = peak_idx
            while right < len(net_counts)-1 and net_counts[right] > threshold:
                right += 1
            n_points = right - left + 1
            area = np.sum(net_counts[left:right]) * angle_step
            widths.append(n_points)
            areas.append(area)
    return peaks, fwhms, areas, widths

# This function generates an annotated plot of the analysis results
def plot_results(angles, raw, bg, net, peak_x, peak_y, labels, fwhms, areas, widths, fwhm_errors, area_errors, gaussian_curves, composition, composition_error, grain_size, grain_size_error):
    """
    Generate annotated plot of XRD analysis results.

    Args:
    angles: Angles data.
    raw: Raw counts data.
    bg: Background radiation values.
    net: Net counts data.
    peak_x: Peak x-coordinates.
    peak_y: Peak y-coordinates.
    labels: Peak labels.
    fwhms: FWHMs of peaks.
    areas: Areas of peaks.
    widths: Widths of peaks.
    fwhm_errors: Errors in FWHMs.
    area_errors: Errors in areas.
    gaussian_curves: Gaussian curves for each peak.
    composition: Composition of the material.
    composition_error: Error in the composition.
    grain_size: Grain size of the material.
    grain_size_error: Error in the grain size.
    """
    #Creates a new figure
    plt.figure(figsize=(12, 7))
    #Plots the raw data
    plt.scatter(angles, raw, s=5, color="orange", label="Raw Data")
    # Plots the background radiation
    plt.scatter(angles, bg, s=5, color="lime", label="Background")
    #Plots the net counts data
    plt.scatter(angles, net, s=5, color="blue", alpha=0.7, label="Background Removed Data")
    # nnotates each peak with its label, FWHM, and area
    for x, y, lbl, fwhm, area, width, fwhm_err, area_err, curve in zip(peak_x, peak_y, labels, fwhms, areas, widths, fwhm_errors, area_errors, gaussian_curves):
        if lbl:
            plt.annotate(f"{lbl}\nFWHM: {fwhm:.2f}±{fwhm_err:.2f}\nArea: {area:.2f}±{area_err:.2f}", (x, y), xytext=(0, 20), textcoords="offset points", ha="center", arrowprops=dict(arrowstyle="-", color="gray"))
        # Plots the Gaussian curve for each peak
        plt.plot(angles, curve, color="red", alpha=0.5)
    # Sets the title and labels
    plt.title("py23sr2")
    plt.xlabel("Diffraction angle 2θ (°)")
    plt.ylabel("Counts")
    #Sets the y-axis to a log scale
    plt.yscale("log")
    # Sets the y-axis limits
    plt.ylim(bottom=1e0) # Set the lower limit of the y-axis to 0.1
    plt.ylim(top=1e3)
    # Sets the x-axis limits
    plt.xlim(left=0)
    #Adds a legend
    plt.legend()
    #Adds a legend for the composition and grain size with 2dp for size
    plt.text(0.02, 0.98, f"Lattice parameter composition: {composition:.2f}±{composition_error:.2f}\nGrain size: {grain_size:.2f}±{grain_size_error:.2f} nm", transform=plt.gca().transAxes, ha="left", va="top", bbox=dict(facecolor='white', alpha=0.8))
    # Adds a grid
    plt.grid(alpha=0.3)
    # Shrinks the layout to reduce overlap fo some of the labels
    plt.tight_layout()

# Main processing pipeline for data analysis
def ProcessData(filename):
    """
    Main processing section for analysis.

    Args:
    filename: Path to the XRD data file.

    Returns:
    Analysis results in a dictionary.
    """
    # Creates the results dictionary
    results = {
        "Peaks": [],
        "Composition": None,
        "Composition_error": None,
        "Grain size": None,
        "Grain size_error": None
    }
    try:
        # Loads the diffraction data from the file from before
        wavelength, angles, counts = load_xrd_data(filename)
        # Calculates the angle step
        angle_step = angles[1] - angles[0]

        # Detects rough peaks in the data
        rough_peaks = detect_peaks(counts, height=150, prominence=20, width=3)
        # Creates a mask excluding regions around the rough peaks
        mask = create_peak_mask(angles, rough_peaks, window=3.0)
        # Fits the background model to the non-peak regions
        bg_params, bg_cov, background = fit_background(angles, counts, mask)
        # Subtracts the background from the counts data
        net_counts = counts - background

        # Performs peak analysis on the background-subtracted data
        final_peaks, fwhms, areas, widths = analyze_peaks(angles, net_counts, height=20, prominence=10, width=2)

        # Sorting the peaks by their angles
        sort_idx = np.argsort(angles[final_peaks])
        sorted_peaks = final_peaks[sort_idx]
        sorted_fwhm = np.array(fwhms)[sort_idx]
        sorted_areas = np.array(areas)[sort_idx]
        sorted_widths = np.array(widths)[sort_idx]

        # Defining the hkl values for the peaks
        hkl = [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0)]
        # Creates labels for the peaks
        labels = [f"({h}{k}{l})" if i<len(hkl) else "" for i, (h,k,l) in enumerate(hkl[:len(sorted_peaks)])]

        #Defines the lattice constants for Cu and Au (From document)
        a_Cu = 361.49e-12
        a_Au = 407.8e-12

        #Initializes lists to store the results
        a_values = []
        a_errors = []
        sizes = []
        size_errors = []
        compositions = []
        comp_errors = []
        # Calculates the standard deviation of the background
        bg_std = np.std(net_counts[mask])

        # Works out the errors in the FWHMs and areas
        fwhm_errors = [angle_step]*len(sorted_fwhm)  # assuming error in FWHM is the same as the angle step
        area_errors = [bg_std * np.sqrt(width) * angle_step for width in sorted_widths]

        # Initializes a list to store the Gaussian curves for each peak
        gaussian_curves = []
        # Looping over all peaks
        for idx, (pk, fwhm, area, width) in enumerate(zip(sorted_peaks, sorted_fwhm, sorted_areas, sorted_widths)):
            # Creates a dictionary to store the peak data
            peak_data = {
                "2theta": float(angles[pk]),
                "2theta_error": angle_step/2,
                "d": None,
                "d_error": None,
                "FWHM": float(fwhm),
                "FWHM_error": angle_step,
                "Area": float(area),
                "Area_error": bg_std * np.sqrt(width) * angle_step
            }

            #Calculates the d-spacing for the peak
            theta = np.deg2rad(angles[pk]/2)
            d = wavelength/(2*np.sin(theta))
            dd = (wavelength * np.cos(theta)/(2*np.sin(theta)**2)) * np.deg2rad(angle_step/2)

            #Stores the d-spacing and its error in the peak data dictionary
            peak_data["d"] = float(d * 1e9)
            peak_data["d_error"] = float(dd * 1e9)

            # Calculates the grain size for the peak
            beta = np.deg2rad(fwhm)
            size = wavelength/(beta*np.cos(theta))
            # Calculates the error in the grain size using differential uncertainty propagation
            size_error = size * np.sqrt((np.deg2rad(angle_step)/beta)**2 + (np.deg2rad(angle_step/2)/np.cos(theta))**2 + (1e-13/wavelength)**2)

            #Checks if the peak has a corresponding hkl value
            if idx < len(hkl):
                h, k, l = hkl[idx]
                # Calculates the lattice constant for the peak
                a = d * np.sqrt(h**2 + k**2 + l**2)
                da = np.sqrt(h**2 + k**2 + l**2) * dd
                # Calculates the composition for the peak
                x = (a - a_Cu)/(a_Au - a_Cu)
                dx = da/(a_Au - a_Cu)

                #Stores the lattice constant, its error, and the composition in the lists
                a_values.append(a)
                a_errors.append(da)
                compositions.append(x)
                comp_errors.append(dx)

            # Stores the grain size and its error in the lists
            sizes.append(size)
            size_errors.append(size_error)
            # Stores the peak data in the results dictionary
            results["Peaks"].append(peak_data)

            # Calculates the Gaussian curve for the peak
            p0 = [net_counts[pk], angles[pk], fwhm/2.355]
            popt, pcov = curve_fit(gaussian, angles, net_counts, p0=p0)
            curve = gaussian(angles, *popt)
            # Stores the Gaussian curve in the list
            gaussian_curves.append(curve)

        # Checks if any compositions were calculated
        if compositions:
            # Calculates the average composition and its error
            x_avg = np.mean(compositions)
            x_err = np.sqrt(sum(e**2 for e in comp_errors))/len(comp_errors)
            # Checks if the average composition is within the valid range
            if 0 <= x_avg <= 1:
                # Stores the average composition and its error in the results dictionary
                results["Composition"] = float(x_avg)
                results["Composition_error"] = float(x_err)
            else:
                print("Warning: Composition is out of range [0, 1].")
        else:
            print("Warning: No composition data available.")

        #checks if any grain sizes were calculated
        if sizes:
            # Calculates the average grain size and its error
            size_avg = np.mean(sizes)
            size_err = np.sqrt(sum(e**2 for e in size_errors))/len(size_errors)
            # Stores the average grain size and its error in the results dictionary
            results["Grain size"] = float(size_avg * 1e9)
            results["Grain size_error"] = float(size_err * 1e9)
        else:
            print("Warning: No grain size data available.")

        # Plots the results
        plot_results(angles, counts, background, net_counts, angles[sorted_peaks], net_counts[sorted_peaks], labels, sorted_fwhm, sorted_areas, sorted_widths, fwhm_errors, area_errors, gaussian_curves, results["Composition"], results["Composition_error"], results["Grain size"], results["Grain size_error"])

    except Exception as e:
        # Prints any errors that occur during processing
        print(f"Error processing data: {str(e)}")
    # Returns the results dictionary
    return results

if __name__ == "__main__":
    # specifies the path to the XRD data file
    data_file = r"C:\Users\remed\Downloads\assessment_data_py23sr2 (1).dat"
    # Processes the data
    output = ProcessData(data_file)
    # Prints out the analysis results
    print("Analysis Results:")
    print(output)
