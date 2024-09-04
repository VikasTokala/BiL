import scipy.io

results = {'-15dB': 0.052896705143233594, '-6dB': 0.024946649930731306, '12dB': 0.01587831084202639, '3dB': 0.017396804201649962, '9dB': 0.016504201223857096, '-20dB': 0.10078691342939226, '20dB': 0.015780970675719343, '60dB': 0.01685680451067988, '0dB': 0.01934388385475905, '15dB': 0.01899435066918647, '30dB': 0.01640727367199576, '25dB': 0.01668888355669776, '-10dB': 0.03122037858720901, '6dB': 0.017344423193167854, '-25dB': 0.1612100452884686, '-3dB': 0.021661288759792483}

# Sort the dictionary based on keys
sorted_results = dict(sorted(results.items(), key=lambda x: int(x[0][:-2])))

# Extract keys and values
keys_list = list(sorted_results.keys())
values_list = [value * 180 for value in sorted_results.values()]  # Multiply each value by 180

# Save keys and values lists to a .mat file
scipy.io.savemat('sorted_results.mat', {'keys_list': keys_list, 'values_list': values_list})
