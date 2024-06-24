# Updates in `functions_ver1.py`
1. 2023-11-11: changed order of args in `cos_i_JN` to `phi_S`, `theta_S`, `phi_J`, `theta_J`
2. 2023-11-11: changed `find_FaceOn_coords`, `find_EdgeOn_coords`, `plot_special_coords` to be more general
3. 2023-11-11: added `cos_i_JN_contour`
4. 2023-11-11: moved `cmd` argument to the first position in `functions`
5. 2023-11-11: renamed `mismatch_epsilon_min_max` function to `optimize_mismatch_gamma_P` and changed its output to dict
6. 2023-11-13: added docstrings to functions
7. 2023-11-19: added `SNR` function
8. 2024-03-17: in Shortcuts section, added `default_plot_fontsizes`, `get_gw`, `get_MLz_from_time_delay`, `get_fcut_from_mcz`, `get_mcz_from_fcut` functions
9. 2024-03-17: edited `SNR` function to accept `f_min`, `delta_f`, `lens_Class`, and `prec_Class` as argument
10. 2024-03-17: in Inclination & Special Coords section, added `cos_i_JN_params` function and updated docstrings for `cos_i_JN`
11. 2024-03-17: added `f_min=20` as default argument for `Sn` function
16. 2024-04-15: added `pycbc.catalog` import

# Updates in `functions_ver2.py`
1. 2023-11-11: changed order of args in `cos_i_JN` to `phi_S`, `theta_S`, `phi_J`, `theta_J`
2. 2023-11-11: changed `find_FaceOn_coords`, `find_EdgeOn_coords`, `plot_special_coords` to be more general
3. 2023-11-11: added `cos_i_JN_contour`
4. 2023-11-11: moved `cmd` argument to the first position in `functions`
5. 2023-11-11: renamed `mismatch_epsilon_min_max` function to `optimize_mismatch_gamma_P` and changed its output to dict
6. 2023-11-13: added docstrings to functions
7. 2023-11-19: added `SNR` function
8. 2024-03-17: in Shortcuts section, added `default_plot_fontsizes`, `get_gw`, `get_MLz_from_time_delay`, `get_fcut_from_mcz`, `get_mcz_from_fcut` functions
9. 2024-03-17: edited `SNR` function to accept `f_min`, `delta_f`, `lens_Class`, and `prec_Class` as argument
10. 2024-03-17: in Inclination & Special Coords section, added `cos_i_JN_params` function and updated docstrings for `cos_i_JN`
11. 2024-03-17: added `f_min=20` as default argument for `Sn` function
12. 2024-03-17: changed `psd_n` to `psd`
13. 2024-03-17: added `psd=None` as default argument for `SNR` function
14. 2024-03-17: improved `mismatch` function to be more generalizable and added `use_opt_match` argument
15. 2024-03-17: added `optimize_missmatch_gammaP`, `optimize_mismatch_mcz`, `find_optimized_coalescence_params` functions
16. 2024-04-15: added `pycbc.catalog` import
17. 2024-04-22: added `number_of_prec_cycles` function
18. 2024-04-22: changed "waveform" to "strain" as a key in the return dictionary of `get_gw` function
19. 2024-04-22: added `plot_waveform_comparison` function
20. 2024-04-22: added `pickle_data` function
21. 2024-04-29: abbreviated `time_delay` to `td` in functions' names
22. 2024-04-29: imported `Classes_ver2` instead of `Classes_ver1`
23. 2024-04-29: changed `Delta_td` to `td` throughout the script
24. 2024-05-02: improved `get_MLz_limits_for_RP_L` function
25. 2024-05-02: added `dir` argument to `pickle_data` function
26. 2024-05-12: renamed `get_MLz_limits_for_RP_L` function to `get_lens_limits_for_RP_L`
27. 2024-05-12: added `number_of_lens_cycles` function
28. 2024-05-13: added `timer_decorator` function
29. 2024-05-22: added `get_y_from_I` function
30. 2024-06-12: added `angle_in_pi_format` function

# Updates in `default_params_ver1.py`
1. 2023-11-10: changed `L` to `J` for `Lensing`, changed `lensing_params_0` to `lens_params_0`
2. 2023-11-13: added `loc_params['sys3']`
3. 2024-01-25: changed `tc` to `t_c`
4. 2024-03-07: added `lens_params_1`, `RP_params_1`, `NP_params_1`

# Updates in `Classes_ver0.py`
1. 2024-02-07: changed `C_amp` in `polarization_amplitude_and_phase` in `Precessing` class to the correct equation (4a in Evangelos's)

# Updates in `Classes_ver1.py`
1. 2023-11-10: changed `L` to `J` for `Lensing` class, capitalized `s` for `theta` & `phi` in `Lensing` class, changed `l_dot_n` to `LdotN`
2. 2023-11-10: deleted `if theta_tilde == 0` under `if cos_i_JN == 1` in the `integrand_delta_phi` function of the `Precessing` class
3. 2024-01-25: changed `tc` to `t_c`
4. 2024-01-25: changed `psi` to `Psi` in `Lensing` class
5. 2024-02-07: changed `C_amp` in `polarization_amplitude_and_phase` in `Precessing` class to the correct equation (4a in Evangelos's)

# Updates in `Classes_ver2.py`
1. 2023-11-10: changed `L` to `J` for `Lensing` class, capitalized `s` for `theta` & `phi` in `Lensing` class, changed `l_dot_n` to `LdotN`
2. 2023-11-10: deleted `if theta_tilde == 0` under `if cos_i_JN == 1` in the `integrand_delta_phi` function of the `Precessing` class
3. 2024-01-25: changed `tc` to `t_c`
4. 2024-01-25: changed `psi` to `Psi` in `Lensing` class
5. 2024-02-07: changed `C_amp` in `polarization_amplitude_and_phase` in `Precessing` class to the correct equation (4a in Evangelos's)
6. 2024-04-29: changed `Delta_td` to `td` in `LensingGeo` class

# Updates in `contours_ver1.py`
1. 2024-04-30: added `get_contours_stats()` function

# Updates in `contours_ver2.py`
1. 2024-04-30: added `get_contours_stats()` function
2. 2024-06-12: added `plot_indiv_contour()` and `plot_indiv_contour_from_dict()` functions
