'''
Author : Maureen Githaiga
Description : Master's Thesis - this script is used to explore the data by plotting the dispersion plots then preprocess the data by 
reducing the dispersion area and smoothing the intensity values and detecting the peaks in the intensity values.
'''
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from scipy.stats import levene
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.signal import savgol_filter,find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,silhouette_score,adjusted_rand_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pybaselines import Baseline


sys.path.append(r'C:\Users\githa\Documents\thesis\Analysis\scripts')

#********************************************DATA*****************************************************************************************************
# Load the data from the CSV files
def load_data(file_path):
    '''
    Load data from a CSV file and return a pandas DataFrame
    :param file_path: the path to the CSV file
    :return: a pandas DataFrame containing the data
    '''
    return pd.read_csv(file_path)

neutral_df = load_data(r"C:\Users\githa\Documents\thesis\Analysis\Data\neutral.csv")
fear_df = load_data(r"C:\Users\githa\Documents\thesis\Analysis\Data\fear.csv")

#removing sample 0 from the data
"""neutral_df = neutral_df.drop(0)
fear_df = fear_df.drop(0)"""

#each index in the dataframe is a dispersion plot which is a matrix where rows are ucv values and columns are usv values
#each element in the matrix is the intensity value
#create those matrices
#ucv (rows) unique values are 120
#usv (columns) unique values are 50
#size of the dispersion plot is 50 by 120

"""for index, row in neutral_df.iterrows():
    dispersion_plot = np.array(ast.literal_eval(row['IntensityTop'])).reshape(50,120)
    normalised_dp = np.apply_along_axis(lambda row : (row - np.mean(row))/np.std(row), 1, dispersion_plot)
    #normalised_dp = np.cbrt(dispersion_plot)
    plt.figure(figsize=(8, 6))
    plt.imshow(normalised_dp)
    plt.show()
    plt.close()"""


def plot_raw_dispersion_plot(dataframe,index,title,ax):
    '''
    Plot dispersion plots for samples in the dataframe
    :param dataframe: the dataframe containing the data
    :param number_samples: the number of samples to plot
    :param title: the title of the plot'''
    #extract the necessary columns
    Ucv = np.array(ast.literal_eval(dataframe.loc[index]['Ucv'])) # x axis values
    Usv = np.array(ast.literal_eval(dataframe.loc[index]['Usv']))# y axis values
    Intensity = np.array(ast.literal_eval(dataframe.loc[index]['IntensityTop'])) #intensity values

    scatter = ax.scatter(Ucv, Usv, c=Intensity, cmap='viridis')
    ax.set_xlabel('Ucv')
    ax.set_title(f'Sweat Sample DMS dispersion plot', fontsize=12, fontweight='bold')
    return scatter



def plot_row_wise_normalised_dispersion_plot(dataframe, index,title,ax):
    '''
    Plot dispersion plots for samples in the dataframe
    :param dataframe: the dataframe containing the data
    :param number_samples: the number of samples to plot
    :param title: the title of the plot'''
    
    Ucv = np.array(ast.literal_eval(dataframe.loc[index]['Ucv'])) # x axis values
    Usv = np.array(ast.literal_eval(dataframe.loc[index]['Usv']))# y axis values
    Intensity = np.array(ast.literal_eval(dataframe.loc[index]['IntensityTop'])).reshape(len(np.unique(Usv)), len(np.unique(Ucv)))#intensity values
    
    #normalise the intensity values
    for i in range (Intensity.shape[0]):
        #Intensity[i] = (Intensity[i] - np.mean(Intensity[i]))/np.std(Intensity[i])
        row = Intensity[i,:]
        #row = (row - np.mean(row))/np.std(row)
        row = np.cbrt(row)
        Intensity[i,:] = row
    
    # Plot the dispersion plot on the given axis
    scatter = ax.scatter(Ucv, Usv, c= Intensity, cmap='viridis')

    #ax.figure.colorbar(scatter, ax=ax, label='Intensity')
    ax.set_xlabel('Ucv')
    #ax.set_ylabel('Separation Voltage')
    ax.set_title(f'Normalised dispersion plot', fontsize=12, fontweight='bold')
    return scatter


#**********************************************************REDUCING DISPERSION PLOT**************************************************************************************************************************************************************
def reduce_dispersion_area(dataframe,Usv_min_threshold,Usv_max_threshold,Ucv_min_threshold,Ucv_max_threshold):
    '''
    '''
    baseline_obj = Baseline()
    filtered_data = []
    for index , row in dataframe.iterrows():
        Ucv = np.array(ast.literal_eval(dataframe.loc[index]['Ucv'])) # x axis values
        Usv = np.array(ast.literal_eval(dataframe.loc[index]['Usv']))# y axis values
        Intensity = np.array(ast.literal_eval(dataframe.loc[index]['IntensityTop'])) #intensity values

        filtered_indices = np.where(((Usv >= Usv_min_threshold)&(Usv <= Usv_max_threshold)) & ((Ucv>= Ucv_min_threshold) & (Ucv <= Ucv_max_threshold)) )
        filtered_Usv = Usv[filtered_indices]
        filtered_Ucv = Ucv[filtered_indices]
        filtered_Intensity = Intensity[filtered_indices]
        

        Intensity = filtered_Intensity.reshape(len(np.unique(filtered_Usv)), len(np.unique(filtered_Ucv)))#filtered intensity values
        #normalise the intensity values
        for i in range (Intensity.shape[0]):
            #Intensity[i] = (Intensity[i] - np.mean(Intensity[i]))/np.std(Intensity[i])
            row = Intensity[i,:]
            row = (row - np.mean(row))/np.std(row)
            #row = np.cbrt(row)
            Intensity[i,:] = row

        flattened_intensity = Intensity.flatten()
        #smoothing
        #smoothing done on the matrix
        Intensity_smoothing_row = savgol_filter(Intensity, 11, 3, axis=0)
        Intensity_smoothing_col = savgol_filter(Intensity_smoothing_row, 11, 3, axis=1)
        flattened_smoothed_intensity = Intensity_smoothing_col.flatten()


        filtered_data.append({
            'index': index,
            'filtered_Ucv': filtered_Ucv.tolist(),
            'filtered_Usv': filtered_Usv.tolist(),
            'normalized_Intensity': flattened_intensity.tolist(),
            'smoothed_Intensity': flattened_smoothed_intensity.tolist(),
            'participants': dataframe.loc[index]['Participants'],
            'Label': dataframe.loc[index]['Label']

        })
    
    # Convert the list of dictionaries into a DataFrame
    filtered_df = pd.DataFrame(filtered_data)
    return filtered_df





def plot_reduced_dispersion_plot(filtered_df,index,plot_type,ax):
        '''
        Plot dispersion plots for samples in the dataframe
        param dataframe: the dataframe containing the data 
        plot_type: the type of plot to display
        
        '''

        data_entry = filtered_df[filtered_df['index'] == index].iloc[0]

        filtered_Ucv = np.array(data_entry['filtered_Ucv'])
        filtered_Usv = np.array(data_entry['filtered_Usv'])
        normalized_Intensity = np.array(data_entry['normalized_Intensity'])
        smoothed_Intensity = np.array(data_entry['smoothed_Intensity'])
         # Reshape the intensity values to their original 2D shape
        unique_Ucv = np.unique(filtered_Ucv)
        unique_Usv = np.unique(filtered_Usv)
        if plot_type == 'non-smoothed':
            Intensity = normalized_Intensity.reshape(len(unique_Usv), len(unique_Ucv))
            tittle = 'Reduced dispersion plot'
        else:
            Intensity = smoothed_Intensity.reshape(len(unique_Usv), len(unique_Ucv))
            tittle = 'Smoothed dispersion plot'

        # Plot the dispersion plot on the given axis
        scatter = ax.scatter(filtered_Ucv, filtered_Usv, c=Intensity, cmap='viridis')
        ax.set_xlabel('Ucv')
        ax.set_ylabel('Usv')
        ax.set_title(tittle,fontsize=12, fontweight='bold')
        return scatter


def check_intensity_range(dataframe):
    '''Check if the intensity values for each sample in the dataframe are between 0 and 1'''
    is_in_range = []
    for index, row in dataframe.iterrows():
        intensity = np.array(row['smoothed_binary_map'])
        if np.all((intensity >= -1) & (intensity <= 1)):
            is_in_range.append(True)
        else:
            is_in_range.append(False)
    return is_in_range

def detect_local_maxima(arr):
    '''
    Detect the local maxima in the given array
    :param arr: the input array
    :return: the indices of the local maxima
    '''
    maxima = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            maxima.append(i)
    return maxima


def detect_peaks(intensity):
    '''
    Detect peaks in the given intensity values
    :param intensity: the intensity values
    :return: the indices of the peaks
    '''
    peaks = []
    for row in intensity:
        row_peaks = detect_local_maxima(row)
        peaks.append(row_peaks)
    return peaks

def binarize_intensity(intensity, peaks):
    binary_map = np.zeros_like(intensity)
    for index, row_peaks in enumerate(peaks):
        for peak in row_peaks:
            binary_map[index, peak] = 1#intensity[index, peak]
    return binary_map

def add_peaks_and_binarise_to_dataframe(dataframe):
    smoothed_peaks_list = []
    non_smoothed_peaks_list = []
    smoothed_binary_map_list = []
    non_smoothed_binary_map_list = []

    for index, row in dataframe.iterrows():
        smoothed_intensity = np.array(row['smoothed_Intensity']).reshape(len(np.unique(row['filtered_Usv'])) , len(np.unique(row['filtered_Ucv'])))
        non_smoothed_intensity = np.array(row['normalized_Intensity']).reshape(len(np.unique(row['filtered_Usv'])) , len(np.unique(row['filtered_Ucv'])))
        
        smoothed_peaks = detect_peaks(smoothed_intensity)
        non_smoothed_peaks = detect_peaks(non_smoothed_intensity)

        smoothed_peaks_list.append(smoothed_peaks)
        non_smoothed_peaks_list.append(non_smoothed_peaks)



        smoothed_binary_map = binarize_intensity(smoothed_intensity, smoothed_peaks)
        non_smoothed_binary_map = binarize_intensity(non_smoothed_intensity, non_smoothed_peaks)
        
        smoothed_binary_map_list.append(smoothed_binary_map)
        non_smoothed_binary_map_list.append(non_smoothed_binary_map)

    dataframe['smoothed_peaks'] = smoothed_peaks_list
    dataframe['non_smoothed_peaks'] = non_smoothed_peaks_list
    dataframe['smoothed_binary_map'] = smoothed_binary_map_list
    dataframe['non_smoothed_binary_map'] = non_smoothed_binary_map_list

    return dataframe

def plot_peaks(dataframe, index):
    row = dataframe.iloc[index]
    smoothed_intensity = np.array(row['smoothed_Intensity']).reshape(len(np.unique(row['filtered_Usv'])), len(np.unique(row['filtered_Ucv'])))
    non_smoothed_intensity = np.array(row['normalized_Intensity']).reshape(len(np.unique(row['filtered_Usv'])), len(np.unique(row['filtered_Ucv'])))
    
    smoothed_peaks = row['smoothed_peaks']
    non_smoothed_peaks = row['non_smoothed_peaks']
    num_rows = smoothed_intensity.shape[0]
    
    plt.figure(figsize=(12, 6))

    # Plot non-smoothed intensity with peaks
    plt.subplot(2, 1, 1)
    i=1
   
    plt.plot(non_smoothed_intensity[i], label=f'Row {i+1}' if i == 0 else "", color='green')
    plt.scatter(non_smoothed_peaks[i], non_smoothed_intensity[i][non_smoothed_peaks[i]], color='red', label='Peaks' if i == 0 else "")
    plt.title('Peaks from raw Intensity Signal')
    plt.legend()
    # Plot smoothed intensity with peaks
    plt.subplot(2, 1, 2)
    plt.plot(smoothed_intensity[i], label=f'Row {i+1}' if i == 0 else "", color='blue')
    plt.scatter(smoothed_peaks[i], smoothed_intensity[i][smoothed_peaks[i]], color='red', label='Peaks' if i == 0 else "")
    plt.title('Peaks from smoothed Intensity Signal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

def plot_dispersion_with_peaks(filtered_df, index):
    row = filtered_df.iloc[index]
    smoothed_intensity = np.array(row['smoothed_Intensity']).reshape(len(np.unique(row['filtered_Usv'])), len(np.unique(row['filtered_Ucv'])))
    non_smoothed_intensity = np.array(row['normalized_Intensity']).reshape(len(np.unique(row['filtered_Usv'])), len(np.unique(row['filtered_Ucv'])))
    
    smoothed_binary_maps = np.array(row['smoothed_binary_map'])
    non_smoothed_binary_maps = np.array(row['non_smoothed_binary_map'])

    filtered_Usv = np.unique(row['filtered_Usv'])
    filtered_Ucv = np.unique(row['filtered_Ucv'])
    
    plt.figure(figsize=(12, 6))
    
    # Plot smoothed intensity with binary map
    plt.imshow(smoothed_intensity, cmap='viridis', aspect='auto', extent=[filtered_Ucv.min(), filtered_Ucv.max(), filtered_Usv.max(), filtered_Usv.min()])
    plt.imshow(smoothed_binary_maps, cmap='grey', alpha=0.5, aspect='auto', extent=[filtered_Ucv.min(), filtered_Ucv.max(), filtered_Usv.max(), filtered_Usv.min()])
    plt.title('Smoothed Dispersion Plot with Detected Peaks',fontsize = 20)
    plt.xlabel('Ucv',fontsize = 18)
    plt.ylabel('Usv',fontsize = 18)
    plt.xticks(fontsize=16)  # Increase the font size of the x-axis tick labels
    plt.yticks(fontsize=16) 
    plt.gca().invert_yaxis()
    #plt.colorbar(label='Intensity')
    
    """ # Plot non-smoothed intensity with binary map
    plt.subplot(2, 1, 2)
    plt.imshow(non_smoothed_intensity, cmap='viridis', aspect='auto', extent=[filtered_Ucv.min(), filtered_Ucv.max(), filtered_Usv.min(), filtered_Usv.max()])
    plt.imshow(non_smoothed_binary_maps, cmap='gray', alpha=0.5, aspect='auto', extent=[filtered_Ucv.min(), filtered_Ucv.max(), filtered_Usv.min(), filtered_Usv.max()])
    plt.title('Binary Map of Non-Smoothed Intensity with Detected Peaks')
    plt.xlabel('Ucv')
    plt.ylabel('Usv')
    plt.colorbar(label='Intensity')
    """
    plt.tight_layout()
    plt.show()

    

def main():
    
    usv_min_threshold = 510
    usv_max_threshold = 722
    ucv_max_threshold = 1.5714
    ucv_min_threshold = -1

    participant_labels = {
    1: [2, 3, 4, 5],
    2: [3, 4, 5, 6],
    3: [4, 5, 6, 7],
    4: [5, 6, 7, 8],
    5: [6, 7, 8, 9],
    6: [7, 8, 9, 11],
    7: [8, 9, 11, 12],
    8: [9, 11, 12, 13],
    9: [11, 12, 13, 14],
    10: [12, 13, 14, 15],
    11: [13, 14, 15, 16],
    12: [14, 15, 16, 17],
    13: [15, 16, 17, 18],
    14: [16, 17, 18, 2],
    15: [17, 18, 2, 3],
    16: [2, 3, 4, 5],
    17: [3, 4, 5, 6],
    18: [4, 5, 6, 7],
    19: [5, 6, 7, 8],
    20: [6, 7, 8, 9],
    21: [7, 8, 9, 11],
    22: [8, 9, 11, 12],
    23: [9, 11, 12, 13],
    24: [11, 12, 13, 14],
    25: [12, 13, 14, 15],
    26: [13, 14, 15, 16],
    27: [14, 15, 16, 17],
    28: [15, 16, 17, 18],
    29: [16, 17, 18, 2],
    30: [17, 18, 2, 3]
    }
    # Adjust participant_labels keys to start from 0
    participant_labels_zero_based = {k-1: v for k, v in participant_labels.items()}
   
    
    fear_df['Label'] = 1  # Label for fear samples
    neutral_df['Label'] = 0  # Label for neutral samples
    
    fear_df['Participants'] = fear_df['Unnamed: 0'].apply(lambda x: participant_labels_zero_based[x])
    neutral_df['Participants'] = neutral_df['Unnamed: 0'].apply(lambda x: participant_labels_zero_based[x]) 
    combined_df = pd.concat([fear_df, neutral_df], ignore_index=True)  # Combine the dataframes

    #preprocess the data by reducing the dispersion area and smoothing the intensity values
    reduced_df = reduce_dispersion_area(combined_df, usv_min_threshold,usv_max_threshold, ucv_min_threshold, ucv_max_threshold)

    #add peaks to the dataframes
    reduced_df = add_peaks_and_binarise_to_dataframe(reduced_df)
    #print(reduced_df['smoothed_binary_map'][0][0])

    #plot peaks and bvinarised peaks
    #plot_peaks(reduced_df, 10)
    plot_dispersion_with_peaks(reduced_df, 1)

    #fig,ax0 = plt.subplots(1,1,figsize=(10,6))
    #fig, (ax0, ax1,ax2,ax3) = plt.subplots(1, 4,figsize =(10,6),layout='constrained')
    #scatter = plot_raw_dispersion_plot(neutral_df, 1, 'Neautral Sample', ax0)
    fig,ax1 = plt.subplots(1,1,figsize=(10,6))
    scatter = plot_row_wise_normalised_dispersion_plot(neutral_df, 1, 'Neutral Sample', ax1)
    #plt.show()
    """#scatter1 = plot_row_wise_normalised_dispersion_plot(fear_df, 12, 'Fear Sample', ax1)
    #scatter1 =plot_raw_dispersion_plot(fear_df, 12, 'Fear Sample', ax1)
    #plot_reduced_dispersion_plot
    scatter = plot_reduced_dispersion_plot(neutral_data, 1, 'Neutral Sample',ax2)
    scatter = plot_smoothed_dispersion_plot(neutral_data, 1, 'Fear Sample',ax3)"""
    """  ax0.set_ylabel('Usv')
    fig.colorbar(scatter, ax=ax0, label='Intensity')
    plt.show()"""


    """#plotting the raw reduced dispersion plot vs the smoothed reduced dispersion plot 
        fig,(ax1,ax2)= plt.subplots(1,2,figsize=(10,6))
        plot_reduced_dispersion_plot(reduced_df, 1,'non-smoothed',ax1)
        scatter = plot_reduced_dispersion_plot(reduced_df, 1, 'smoothed',ax2)
        fig.colorbar(scatter, ax=ax2, label='Intensity')
        plt.show()
    """
    print(reduced_df.columns)
    #save preproicessed data to a csv file                                                            
    reduced_df.reset_index(drop=True, inplace=True)
    reduced_df.to_csv(r"C:\Users\githa\Documents\thesis\Analysis\Data\preprocessed_data_1.csv", index=False)                                                                                                                                                                                                                                                    

    # check if the intensity values are between -1 and 1
    #is_in_range = check_intensity_range(preprocessed_df)
    #print(f'All intensity values are between -1 and 1: {all(is_in_range)}')

    


if __name__ == "__main__":
    main()