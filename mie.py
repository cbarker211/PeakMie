import numpy as np
from matplotlib import pyplot as plt


def cauchy(wlen, A, B, C):
    n = A + (B / (wlen ** 2)) + (C / (wlen ** 4))
    return n


def rel_ref_index(n_particle, n_medium):
    return n_particle / n_medium


def calc_size_param(radius, wavelength):
    return (2 * np.pi * radius) / (wavelength / 1.00027)


def log_determinant(size, m, x):
    """
    size :: integer, size of array [nmx in legacy code]
    m :: float, relative refractive index
    x :: float, size parameter
    """
    D = np.zeros(size, dtype=np.complex128)
    for n in range(size - 1, 0, -1):
        d = n / (m * x)
        D[n - 2] = d - (1 / (D[n - 1] + d))
    return D


def cone_angle(start_angle, end_angle, intensities_found):
    """
    Take out elements of the array found between specified angles
    Assumes intensities_found is between 0 and pi radians
    start_angle :: float
    end_angle :: float
    intensities_found :: [float]
    """
    if (start_angle < 0) or (end_angle > np.pi) or (start_angle > end_angle):
        raise ValueError("Bad angle bounds")

    angle_per_element = np.pi / np.size(intensities_found)

    return intensities_found[
        np.int_(start_angle / angle_per_element) : np.int_(end_angle / angle_per_element)
    ]


def average_peak_diff(x, y):
    """
    Takes in two arrays and returns the average difference between corresponding elements.
    """
    dif = np.subtract(x, y)
    return np.sum(np.abs(dif)) / len(dif)


# bhmie outputs the cross sections, and also an array of intensity at each angle
def bhmie(
    size_param, n_particle, n_medium, num_angles_in_range, start_angle, end_angle
):
    ref_index = rel_ref_index(n_particle, n_medium)
    if end_angle > 90:
        backscatter = True

    # calc number of angles over full range
    num_angles = (
        np.int_(np.ceil(num_angles_in_range * (90 / (end_angle - start_angle)))) + 1
    )

    # Number of angles needs to be between 2 and 1000
    if num_angles > 1000:
        num_angles = 1000
    elif num_angles < 1:
        num_angles = 1

    #  number of terms that need to be summed
    max_terms = 15000000

    # Create arrays full of zeroes, which need to be filled
    s1_1 = np.zeros(num_angles, dtype=np.complex128)
    s1_2 = np.zeros(num_angles, dtype=np.complex128)
    s2_1 = np.zeros(num_angles, dtype=np.complex128)
    s2_2 = np.zeros(num_angles, dtype=np.complex128)
    pi = np.zeros(num_angles, dtype=np.complex128)
    tau = np.zeros(num_angles, dtype=np.complex128)

    # Series expansion terminated after nstop terms
    # Logarithmic derivatives calculated from nmx on down
    nstop = np.int_(size_param + 4.05 * size_param ** (1 / 3.0) + 2.0)

    # TODO: where does this come from??
    nmx = max(nstop, abs(ref_index * size_param)) + 15.0

    if nmx > max_terms:
        max_terms = nmx
        # print("error: nmx > nmxx=%f for |m|x=%f" % (max_terms, abs(size_param * ref_index)))
        return

    # Create array of mu's where mu = cos(theta)
    # Only for angles we want to look at - given at start
    # angle_subtended = end_angle - start_angle
    size_angle = (np.pi * 0.5) / (num_angles - 1)  # Size of each angle
    theta = np.arange(0, num_angles, 1)
    mu = np.cos(theta * size_angle)

    # Iniialize iteration variables
    pi0 = np.zeros(num_angles, dtype=np.complex128)
    pi1 = np.ones(num_angles, dtype=np.complex128)

    # Logarithmic derivative D(J) calculated by downward recurrence beginning with initial value Jn = nmx
    D = log_determinant(int(nmx), ref_index, size_param)

    # Riccati-Bessel functions with real argument X calculated by upward recurrence

    psi0 = chi1 = np.cos(size_param)
    psi1 = np.sin(size_param)
    chi0 = -np.sin(size_param)
    xi1 = psi1 - chi1 * 1j
    scattering_eff = 0
    asymmetry_param = 0
    p = -1

    an = 0.0
    bn = 0.0

    for n in range(0, nstop):
        en = n + 1.0
        fn = (2.0 * en + 1.0) / (en * (en + 1.0))

        # Calculate psi_n and chi_n
        psi = (2.0 * en - 1.0) * psi1 / size_param - psi0
        chi = (2.0 * en - 1.0) * chi1 / size_param - chi0
        xi = psi - chi * 1j

        # *** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
        if n > 0:
            an1 = an
            bn1 = bn

        # Compute AN and BN:
        an = ((D[n] / ref_index + en / size_param) * psi - psi1) / (
            (D[n] / ref_index + en / size_param) * xi - xi1
        )
        bn = ((ref_index * D[n] + en / size_param) * psi - psi1) / (
            (ref_index * D[n] + en / size_param) * xi - xi1
        )

        # Augment sums for scattering_eff and g=<cos(theta)>
        scattering_eff += (2.0 * en + 1.0) * (np.abs(an) ** 2 + abs(bn) ** 2)
        asymmetry_param += ((2.0 * en + 1.0) / (en * (en + 1.0))) * (
            np.real(an) * np.real(bn) + np.imag(an) * np.imag(bn)
        )

        # if (n > 0):
        #     asymmetry_param += ((en-1.)* (en+1.)/en)*( np.real(an1)* real(an)+imag(an1)*imag(an)+real(bn1)* real(bn)+imag(bn1)*imag(bn))
        #

        # Now calculate scattering intensity pattern
        pi = pi1  # pi1 because we want a hard copy of the values
        tau = en * mu * pi - (en + 1.0) * pi0
        if backscatter == False:
            s1_1 += fn * (an * pi + bn * tau)
            s2_1 += fn * (an * tau + bn * pi)
            p = -p
        else:
            p = -p
            s1_2 += fn * p * (an * pi - bn * tau)
            s2_2 += fn * p * (bn * pi - an * tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1 - chi1 * 1j

        # Compute pi_n for next value of n
        # For each angle J, compute pi_n+1 from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.0 * en + 1.0) * mu * pi - (en + 1.0) * pi0) / en
        pi0 = 0 + pi  # 0+pi because we want a hard copy of the values

    # s1=np.concatenate((s1_1,s1_2[-2::-1]))
    # s2=np.concatenate((s2_1,s2_2[-2::-1]))
    asymmetry_param = 2.0 * asymmetry_param / scattering_eff
    # scattering_eff = (2./(size_param*size_param))*scattering_eff
    # extinction_eff = (4./(size_param*size_param))*np.real(s1[0])
    # backscatter_eff = 4*(abs(s1[2*num_angles-2])/size_param)**2

    # intensity= (np.sum(np.abs(s1))**2 + np.sum(np.abs(s2))**2) # I= sqrt(|S1|^2 + |S2|^2)
    # intensities = (np.abs(s1))**2 + (np.abs(s2))**2
    if backscatter == True:
        # print(s1_2)
        s1 = np.sum((np.abs(s1_2[-2::-1])) ** 2)
        s2 = np.sum((np.abs(s2_2[-2::-1])) ** 2)
        # comment out either half of the next line if you want paralell or perpendicular polarization
        intensities = (np.abs(s2_2[-2::-1])) ** 2 + (np.abs(s1_2[-2::-1])) ** 2
        # inten_not_squared = (np.abs(s1_2[-2::-1])) + (np.abs(s2_2[-2::-1]))
        start_index = np.int_(((start_angle - 90) / 90) * num_angles)
        end_index = np.int_(((end_angle - 90) / 90) * num_angles)
        # print(len(intensities))
        # print(np.linspace(start_index, end_index, 62))
        intensities = intensities[start_index:end_index]
        # print(intensities)
        # inten_not_squared = inten_not_squared[start_index:end_index]
        # s1= (s1_2[-2::-1][start_index:end_index])
        # s1_imag = np.array(s1_2.imag[-2::-1][start_index:end_index])**2
        # s2 = (s2_2[-2::-1][start_index:end_index])
    else:
        s1 = np.abs(s1_1) ** 2
        s2 = np.abs(s2_1) ** 2
        # comment out either half of the next line if you want paralell or perpendicular polarization
        intensities = np.abs(s1_1) ** 2 + (np.abs(s2_1) ** 2)
        start_index = np.int_(np.floor(((start_angle) / 90) * num_angles))
        end_index = np.int_(np.ceil(((end_angle) / 90) * num_angles))
        intensities = intensities[start_index:end_index]
    intensity = np.sum(intensities)
    # inten_nsq = np.sum(inten_not_squared)

    return intensity
