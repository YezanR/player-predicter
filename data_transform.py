import numpy 
import pandas 
import math
import copy

def transformTestData(dataFrame):
    dataFrame = copy.copy(dataFrame)
     # Filtrer les données dont on en a pas besoin
    del dataFrame['Name']
    del dataFrame['Photo']
    del dataFrame['Flag']
    del dataFrame['Real Face']
    del dataFrame['Joined']
    del dataFrame['Jersey Number']
    del dataFrame['Weight']
    del dataFrame['Height']
    del dataFrame['Preferred Foot']
    del dataFrame['Contract Valid Until']
    del dataFrame['Club Logo']
    del dataFrame['Work Rate']
    del dataFrame['Loaned From']
    del dataFrame['Price']

    # remove € character, leave just numbers
    dataFrame['Wage'] = dataFrame['Wage'].str.replace('€', '')
    dataFrame['Release Clause'] = dataFrame['Release Clause'].str.replace('€', '')

    # get rid of any empty values for the column Release Clause
    dataFrame['Release Clause'].replace('', numpy.nan, inplace=True)
    dataFrame.dropna(subset=['Release Clause'], inplace=True)

    # We get the actual value of the columns with the format 'number'+'digit' 
    positionCols = ['LS', 'ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']
    for pos in positionCols:
        dataFrame[pos] = dataFrame[pos].apply(lambda x: evalPositionScore(x))

    # Applying the function parseValue to the following columns
    dataFrame['Release Clause'] = dataFrame['Release Clause'].apply(lambda x: parseValue(x))
    dataFrame['Wage'] = dataFrame['Wage'].apply(lambda x: parseValue(x))

    return dataFrame


def transformLearningData(dataFrame):
    dataFrame = transformTestData(dataFrame)

    dataFrame['Value'] = dataFrame['Value'].str.replace('€', '')
    dataFrame['Value'] = dataFrame['Value'].apply(lambda x: parseValue(x))
    return dataFrame


def extractFeatures(dataFrame):
    # Remplaçons les données par les one-hot encoded data
    features = pandas.get_dummies(dataFrame, columns=['Nationality', 'Club', 'Body Type', 'Position',])

    # On enlève les valeurs " Price " et " Value " car ce sont les valeurs qu'on cherche
    del features['Value']

    features.fillna(dataFrame.mean(), inplace=True)

    return features


# parse string for millions and thousands to numeric values
def parseValue(strVal):
    if 'M' in strVal:
        value = strVal.replace('M', '')
        return int(float(value) * 1000000)
    elif 'K' in strVal:
        return int(float(strVal.replace('K', '')) * 1000)
    else:
        return int(strVal)

def evalPositionScore(position):
    if pandas.isnull(position):
        return 0
    else:
        return eval(position)

    