"""
XRD Data Analysis Module.

This module provides functions to load, clean, analyze, and visualize
X-ray diffraction (XRD) data, including peak detection and composition analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths


def load_xrd_data(file_path):
    """
    Load XRD data from file and extract experimental parameters.

    Args:
        file_path (str): Path to the XRD data file.

    Returns:
        tuple: Wavelength, angles, and counts.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        wavelength = None
        
        for line in lines:
            if line.startswith("Wavelength"):
                wavelength = float(line.split("=")[1].strip()) * 1e-10
                break
                
        if wavelength is None:
            raise ValueError("Wavelength not found in file header")

    skip_lines = lines.index(next(line for line in lines if "&END" in line)) + 2
    angles, counts = np.genfromtxt(
        file_path, skip_header=skip_lines, unpack=True
    )
    return wavelength, angles, counts


def detect_peaks(counts, height=150, prominence=20, width=3):
    """
    Initial peak detection using conservative thresholds.

    Args:
        counts: Counts data.
        height: Height threshold.
        prominence: Prominence threshold.
        width: Width threshold.

    Returns:
        Indices of detected peaks.
    """
    peaks, _ = find_peaks(
        counts, height=height, prominence=prominence, width=width
    )
    return peaks


def background_model(two_theta, base, center):
    """
    Quadratic model for X-ray background radiation.

    Args:
        two_theta: Angles data.
        base: Base parameter.
        center: Center parameter.

    Returns:
        Background radiation values.
    """
    return base - ((two_theta - center) ** 2) / 100


def create_peak_mask(angles, peak_indices, window=3.0):
    """
    Create mask excluding regions around detected peaks.

    Args:
        angles: Angles data.
        peak_indices: Indices of detected peaks.
        window: Window size.

    Returns:
        Mask values.
    """
    angle_step = angles[1] - angles[0]
    mask = np.ones_like(angles, dtype=bool)
    exclude_points = int(window / (2 * angle_step))
    
    for idx in peak_indices:
        start = max(0, idx - exclude_points)
        end = min(len(angles), idx + exclude_points)
        mask[start:end] = False
        
    return mask


def fit_background(angles, counts, mask):
    """
    Fit background model to non-peak regions.

    Args:
        angles: Angles data.
        counts: Counts data.
        mask: Mask values.

    Returns:
        tuple: Background parameters, covariance matrix, and curve.
    """
    p0_params = [np.median(counts[mask]), np.median(angles[mask])]
    popt, pcov = curve_fit(
        background_model, angles[mask], counts[mask], p0=p0_params
    )
    bg_curve = background_model(angles, *popt)
    return popt, pcov, bg_curve


def gaussian(x_val, amplitude, center, sigma):
    """
    Gaussian function.

    Args:
        x_val: Input values.
        amplitude: Amplitude.
        center: Center.
        sigma: Standard deviation.

    Returns:
        Gaussian values.
    """
    return amplitude * np.exp(-(x_val - center) ** 2 / (2 * sigma ** 2))


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
        tuple: Peaks, FWHMs, areas, and widths.
    """
    peaks, _ = find_peaks(
        net_counts, height=height, prominence=prominence, width=width
    )
    
    fwhms = []
    areas = []
    widths = []
    angle_step = angles[1] - angles[0]
    
    if len(peaks) > 0:
        width_data = peak_widths(net_counts, peaks, rel_height=0.5)
        fwhms = width_data[0] * angle_step
        
        for peak_idx in peaks:
            peak_height = net_counts[peak_idx]
            threshold = 0.05 * peak_height
            left = peak_idx
            
            while left > 0 and net_counts[left] > threshold:
                left -= 1
                
            right = peak_idx
            while right < len(net_counts) - 1 and net_counts[right] > threshold:
                right += 1
                
            n_points = right - left + 1
            area = np.sum(net_counts[left:right]) * angle_step
            widths.append(n_points)
            areas.append(area)
            
    return peaks, fwhms, areas, widths


# pylint: disable=too-many-arguments, too-many-locals
def plot_results(angles, raw, bg_curve, net, peak_x, peak_y, labels, 
                 fwhms, areas, widths, fwhm_errors, area_errors, 
                 gaussian_curves, composition, composition_error, 
                 grain_size, grain_size_error):
    """
    Generate annotated plot of XRD analysis results.

    Args:
        angles: Angles data.
        raw: Raw counts data.
        bg_curve: Background radiation values.
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
    plt.figure(figsize=(12, 7))
    plt.scatter(angles, raw, s=5, color="orange", label="Raw Data")
    plt.scatter(angles, bg_curve, s=5, color="lime", label="Background")
    plt.scatter(angles, net, s=5, color="blue", alpha=0.7, 
                label="Background Removed Data")
                
    for x_val, y_val, lbl, fwhm, area, _, fwhm_err, area_err, curve in zip(
            peak_x, peak_y, labels, fwhms, areas, widths, fwhm_errors, 
            area_errors, gaussian_curves):
        if lbl:
            annotation_text = (f"{lbl}\nFWHM: {fwhm:.2f}±{fwhm_err:.2f}\n"
                               f"Area: {area:.2f}±{area_err:.2f}")
            plt.annotate(
                annotation_text, (x_val, y_val), xytext=(0, 20), 
                textcoords="offset points", ha="center", 
                arrowprops=dict(arrowstyle="-", color="gray")
            )
        plt.plot(angles, curve, color="red", alpha=0.5)
        
    plt.title("py23sr2")
    plt.xlabel("Diffraction angle 2θ (°)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.ylim(bottom=1e0) 
    plt.ylim(top=1e3)
    plt.xlim(left=0)
    plt.legend()
    
    stat_text = (f"Lattice parameter composition: {composition:.2f}±"
                 f"{composition_error:.2f}\n"
                 f"Grain size: {grain_size:.2f}±{grain_size_error:.2f} nm")
    plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes, 
             ha="left", va="top", bbox=dict(facecolor='white', alpha=0.8))
             
    plt.grid(alpha=0.3)
    plt.tight_layout()


# pylint: disable=too-many-locals, too-many-statements, broad-except
def process_data(filename):
    """
    Main processing section for analysis.

    Args:
        filename: Path to the XRD data file.

    Returns:
        Analysis results in a dictionary.
    """
    results = {
        "Peaks": [],
        "Composition": None,
        "Composition_error": None,
        "Grain size": None,
        "Grain size_error": None
    }
    
    try:
        wavelength, angles, counts = load_xrd_data(filename)
        angle_step = angles[1] - angles[0]

        rough_peaks = detect_peaks(counts, height=150, prominence=20, width=3)
        mask = create_peak_mask(angles, rough_peaks, window=3.0)
        _, _, background = fit_background(angles, counts, mask)
        net_counts = counts - background

        final_peaks, fwhms, areas, widths = analyze_peaks(
            angles, net_counts, height=20, prominence=10, width=2
        )

        sort_idx = np.argsort(angles[final_peaks])
        sorted_peaks = final_peaks[sort_idx]
        sorted_fwhm = np.array(fwhms)[sort_idx]
        sorted_areas = np.array(areas)[sort_idx]
        sorted_widths = np.array(widths)[sort_idx]

        hkl = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0)]
        labels = [
            f"({h_val}{k_val}{l_val})" if i < len(hkl) else "" 
            for i, (h_val, k_val, l_val) in enumerate(hkl[:len(sorted_peaks)])
        ]

        lattice_cu = 361.49e-12
        lattice_au = 407.8e-12

        a_values = []
        a_errors = []
        sizes = []
        size_errors = []
        compositions = []
        comp_errors = []
        
        bg_std = np.std(net_counts[mask])
        fwhm_errors = [angle_step] * len(sorted_fwhm)
        area_errors = [
            bg_std * np.sqrt(width) * angle_step for width in sorted_widths
        ]

        gaussian_curves = []
        
        for idx, (peak_idx, fwhm, area, width) in enumerate(
                zip(sorted_peaks, sorted_fwhm, sorted_areas, sorted_widths)):
            
            peak_data = {
                "2theta": float(angles[peak_idx]),
                "2theta_error": angle_step / 2,
                "d": None,
                "d_error": None,
                "FWHM": float(fwhm),
                "FWHM_error": angle_step,
                "Area": float(area),
                "Area_error": bg_std * np.sqrt(width) * angle_step
            }

            theta = np.deg2rad(angles[peak_idx] / 2)
            d_spacing = wavelength / (2 * np.sin(theta))
            d_spacing_err = (
                (wavelength * np.cos(theta) / (2 * np.sin(theta) ** 2)) * 
                np.deg2rad(angle_step / 2)
            )

            peak_data["d"] = float(d_spacing * 1e9)
            peak_data["d_error"] = float(d_spacing_err * 1e9)

            beta = np.deg2rad(fwhm)
            size = wavelength / (beta * np.cos(theta))
            
            err_term_1 = (np.deg2rad(angle_step) / beta) ** 2
            err_term_2 = (np.deg2rad(angle_step / 2) / np.cos(theta)) ** 2
            err_term_3 = (1e-13 / wavelength) ** 2
            size_error = size * np.sqrt(err_term_1 + err_term_2 + err_term_3)

            if idx < len(hkl):
                h_val, k_val, l_val = hkl[idx]
                hkl_sum_sq = h_val ** 2 + k_val ** 2 + l_val ** 2
                
                lattice_const = d_spacing * np.sqrt(hkl_sum_sq)
                lattice_err = np.sqrt(hkl_sum_sq) * d_spacing_err
                
                comp_x = (lattice_const - lattice_cu) / (lattice_au - lattice_cu)
                comp_dx = lattice_err / (lattice_au - lattice_cu)

                a_values.append(lattice_const)
                a_errors.append(lattice_err)
                compositions.append(comp_x)
                comp_errors.append(comp_dx)

            sizes.append(size)
            size_errors.append(size_error)
            results["Peaks"].append(peak_data)

            p0_params = [net_counts[peak_idx], angles[peak_idx], fwhm / 2.355]
            popt, _ = curve_fit(gaussian, angles, net_counts, p0=p0_params)
            curve = gaussian(angles, *popt)
            gaussian_curves.append(curve)

        if compositions:
            x_avg = np.mean(compositions)
            x_err = np.sqrt(sum(err ** 2 for err in comp_errors)) / len(comp_errors)
            if 0 <= x_avg <= 1:
                results["Composition"] = float(x_avg)
                results["Composition_error"] = float(x_err)
            else:
                print("Warning: Composition is out of range [0, 1].")
        else:
            print("Warning: No composition data available.")

        if sizes:
            size_avg = np.mean(sizes)
            size_err = np.sqrt(sum(err ** 2 for err in size_errors)) / len(size_errors)
            results["Grain size"] = float(size_avg * 1e9)
            results["Grain size_error"] = float(size_err * 1e9)
        else:
            print("Warning: No grain size data available.")

        plot_results(
            angles, counts, background, net_counts, 
            angles[sorted_peaks], net_counts[sorted_peaks], 
            labels, sorted_fwhm, sorted_areas, sorted_widths, 
            fwhm_errors, area_errors, gaussian_curves, 
            results["Composition"], results["Composition_error"], 
            results["Grain size"], results["Grain size_error"]
        )

    except Exception as error:
        print(f"Error processing data: {str(error)}")
        
    return results


if __name__ == "__main__":
    DATA_FILE = r"C:\Users\remed\Downloads\assessment_data_py23sr2 (1).dat"
    OUTPUT = process_data(DATA_FILE)
    print("Analysis Results:")
    print(OUTPUT)