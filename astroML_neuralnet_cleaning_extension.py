'''
package by Juan Sebastian Lozano

This module is an extention to the astroML package which allows for easy data cleaning and 
matching for the SDSS spectra data set and the SEGUE extended data set

This module's main use case is creating ML ready data from the raw spectra corresponding to stars in the SEGUE catalog. 

Credit to the astroML package by Jake Vanderplas and "Statistics, Data Mining, and 
Machine Learning in Astronomy" by Zeljko Ivezic, Andrew Connolly,
Jacob Vanderplas, and Alex Gray
'''

#Machine Learning and Data Libraries
from astroML.datasets import fetch_sdss_spectrum        # data importer for SDSS (http://www.sdss3.org/dr9/) information
from astroML.datasets import fetch_sdss_sspp            # data set importer for SEGUE Stellar Parameters Pipeline Parameters

#Computation Libraries 
import numpy as np                                      # array processing 
import matplotlib.pyplot as plt                         # plotting 
import pandas                                           # data processing in r-like data frams44

#Control Flow Libraries
from urllib2 import HTTPError                           # allows http error handling when loading data


''' Functions  '''

'''
# gets and caches fetch_sdss_sspp
# input: none
# output: pandas DataFrame with all of the sspp data 
'''    
def getSpectrumData():
    #SEGUE provides big-endian data, pandas only has small-endian capabilities, so .newbyteorder().byteswap() covert one to the other
    return pandas.DataFrame(fetch_sdss_sspp().newbyteorder().byteswap())

'''
# Plots the spectra of a sample of stars,
# input: a DataFrame of values from fetch_sdss_sspp
# output: no return, displays a matplotlib object
'''
def plotSampleSpectra(sample_spectra):

    #get the array of plate, mjd, fiber values for each star in sample 
    #recall that fetch_sdss_spectrum() idenifies uniquely the star by these three values
    sample_spectra_plate = sample_spectra["plate"]
    sample_spectra_mjd = sample_spectra["mjd"]
    sample_spectra_fiber = sample_spectra["fiber"]
    
    #add the spectra of each star to a plt object, then show the plot 
    for star_index in sample_spectra.index:
        
        plate = sample_spectra_plate[star_index]
        mjd = sample_spectra_mjd[star_index]
        fiber = sample_spectra_fiber[star_index]
        
        print(plate , mjd , fiber)
        
        #try-except is here to handle http errors in getting the spectra
        try:
            spectrum = fetch_sdss_spectrum(plate , mjd , fiber)
            
            fig, ax = plt.subplots()
            ax.plot(spectrum.wavelength(), spectrum.spectrum, '-k')
            title = "Spectra of (" +str(plate) + "," + str(mjd) + "," + str(fiber) + ")"
            ax.set(xlabel = "Wavelength (Angstroms)", ylabel = r'Magnitude $(10^{-7} erg/(A s cm^2)$', title = title)
            ax.grid()
            fig.savefig("./plots/" + title + ".png")
            plt.close(fig)
        except HTTPError:
            print(plate , mjd , fiber, "Not Found")
        file_name = "Spectra_of_" +str(plate) + "_" + str(mjd) + "_" + str(fiber)
        

'''    
# Fetches the spectra from fetch_sdss_spectrum for the given stars
# input: a DataFrame of values from fetch_sdss_sspp
# output: a list of the form 
#               [ (StarIndex (matching that of the index of the given star in sample_spectra),
#                   [(wavelength, magnitude)]) ]
'''
def returnSampleSpectra(sample_spectra):
    sample_spectra_plate = sample_spectra["plate"]
    sample_spectra_mjd = sample_spectra["mjd"]
    sample_spectra_fiber = sample_spectra["fiber"]
    
    sample_spectra_data = []
    
    #loop through every given star in the sample, add thir spectra data to sample_spectra_data 
    #this loops gives a three tuple for each star (starID, [list of wavelengths],[list of magnitudes])
    for star_index in sample_spectra.index:
        
        plate = sample_spectra_plate[star_index]
        mjd = sample_spectra_mjd[star_index]
        fiber = sample_spectra_fiber[star_index]
                
        #try-except is here to handle http errors in getting the spectra
        try:
            SDSS_object = fetch_sdss_spectrum(plate , mjd , fiber)
            sample_spectra_data.append([star_index,SDSS_object.wavelength(), SDSS_object.spectrum])
        except HTTPError:
            print(plate , mjd , fiber, "Not Found")
        else:
            print(plate , mjd , fiber, "MISC ERROR")
    #make into pandas DataFrame to make manipulation easier
    sample_spectra_data_frame = pandas.DataFrame(sample_spectra_data)
    
    #We will now make it into a list new_spectra of two-tuples [(starID,[(wavelength,magnitude)])]
    new_spectra = []
    
    #loop through each star and make a list star_data of (wavelength,magnitude)
    for star in xrange(0,len(sample_spectra_data_frame)-1):
        star_data = []
        for i in xrange(0,len(sample_spectra_data_frame[1][star])-1):
            star_data.append((sample_spectra_data_frame[1][star][i],sample_spectra_data_frame[2][star][i]))
        
        new_spectra.append((sample_spectra_data_frame.iloc[star][0],star_data))
    
    return new_spectra

'''    
# Takes the output from returnSampleSpectra and averages it by bins and returns the averages, meant as training data
# input: output of returnSampleSpectra, int of the number of bins wanted
# output: a list of bins (lower bounds for binds), the size of each bin
'''
def returnSpectraBins(sample_spectra_data, bin_number):
    
    #We will calculate the range of the wavelengths by disaggregating the wavelengths and then taking their range
    range = []
    for star in xrange(0,len(sample_spectra_data)-1):
        sample_spectra_data[star][0]
        for i in sample_spectra_data[star][1]:
            range.append(i[0])
    #bin_size = range/bin_number        
    bin_size = (np.max(range)-np.min(range))/bin_number
    
    #list of lower bounds of the bins
    bins = [np.min(range)+bin_size*i for i in xrange(0,bin_number)]
    return bins, bin_size

'''    
# Takes the data from returnSampleSpectra and bins it according to returnSpectraBins
# input: output of returnSampleSpectra, the output of returnSpectraBins
# output: a list of the form 
#             [(starIndex, [average magnitude for each bin])]
'''    
def binData(sample_spectra_data, bins, size):
    
    new_spectra_data = []
    
    #This loop will bin and average the data for each star 
    for star_index in xrange(0,len(sample_spectra_data)-1):
        
        star_data = []
        
        #we loop through each bin and add the points in the bin into a "points_to_be_averaged" list, then average them
        for lower_bound in bins:
            points_to_be_averaged = []
            for point in sample_spectra_data[star_index][1]:
                if point[0] <= lower_bound + size and point[0] >= lower_bound:
                    points_to_be_averaged.append(point[1])
            
            star_data.append(np.mean(points_to_be_averaged))
        #each star now has a tuple (starIndex, [average magnitude for each bin])
        new_spectra_data.append((sample_spectra_data[star_index][0],star_data))

    return new_spectra_data
    
