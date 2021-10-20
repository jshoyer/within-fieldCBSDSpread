# -*- coding: utf-8 -*-
from __future__ import division

import matplotlib
matplotlib.use('Agg')

import os
import sys
import random
import math
import numpy as np
import pandas as pd
import bisect
import json
import csv
import pdb
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

def simulation(argv):    
    def prime_factors(n): #(code from stackexchange)
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors
    
    def find_gt(a, x): #'Find leftmost value greater than x' (code from python documentation)
        i = bisect.bisect_right(a, x)
        if i != len(a):
            return i #returns index, not value

    def surveyPlants(currentTime, cassavaPop, plantsSurveyFn, accuracySurveyFn): #how to do surveys
        survey = []
        numBoxes = plantsSurveyFn/2.0
    
        for i in range(int(numBoxes)):
            #survey bottom L to top R transect
            xVal = random.randrange(int(i*xdim/numBoxes), int((i+1)*xdim/numBoxes))
            yVal = random.randrange(int(i*ydim/numBoxes), int((i+1)*ydim/numBoxes))
            surveyedPlant = cassavaPop.loc[(cassavaPop['xPos'] == xVal) & (cassavaPop['yPos'] == yVal)].index.tolist()
            if cassavaPop['Infected'][surveyedPlant[0]] == 1 and cassavaPop['SymptomDate'][surveyedPlant[0]] < currentTime and cassavaPop['Symptoms'][surveyedPlant[0]]*accuracySurveyFn > random.random():
                survey.append(1)
            else:
                survey.append(0)
    
            #survey bottom top L to bottom R transect
            xVal = random.randrange(int(i*xdim/numBoxes), int((i+1)*xdim/numBoxes))
            yVal = random.randrange(int(ydim - (i+1)*ydim/numBoxes), int(ydim - i*ydim/numBoxes))
            surveyedPlant = cassavaPop.loc[(cassavaPop['xPos'] == xVal) & (cassavaPop['yPos'] == yVal)].index.tolist()
            if cassavaPop['Infected'][surveyedPlant[0]] == 1 and cassavaPop['SymptomDate'][surveyedPlant[0]] < currentTime and cassavaPop['Symptoms'][surveyedPlant[0]]*accuracySurveyFn > random.random():
                survey.append(1)
            else:
                survey.append(0)
    
        return sum(survey)/float(plantsSurveyFn)*popSize
                
    def roguing(cassavaPop, probList): #how to rogue plants
        removedPlants = []
        for plant in range(len(cassavaPop)):
            if cassavaPop['Symptoms'][plant]*roguingAccuracy > random.random(): #don't rogue the same plant twice  
                cassavaPop['livePlants'][plant] = 0
                removedPlants.append(plant)
                
                #update settings in cassavaPop for the rogued plant
                cassavaPop['Infected'][plant] = 0 #stop adding the plant to the infection totals
                cassavaPop['Symptoms'][plant] = 0 #impossible to rougue plant again
                cassavaPop['Whitefly'][plant] = 0 #all whitefly on the plant dissapear
                cassavaPop['Proportion'][plant] = 0  
    
                #prevent whitefly from going to the rogued plant
                distanceMatrix[:,plant] = 0
        
        #regenerate probabilities from the distanceMatrix 
        probList = np.zeros((popSize,popSize))        
        scale = np.linalg.norm(distanceMatrix, 1, axis = 1)
        for row in range(len(distanceMatrix)):
            if scale[row] != 0:
                distanceMatrix[row] /= scale[row]
            probList[row] = np.cumsum(distanceMatrix[row])
            
        return removedPlants, probList
    
    
    def growingSeason(currentTime, probList, initialInfection = None, initialField = None):
        print 'year', year
        
        #Initialise the field:
        if year == endYear - 1: #if this is the last year
            plantsToReplantField = int(popSize/multiplicationRatio[0])
        else: #take into account the clean seed input from the next year when deciding if a field is dead 
            plantsToReplantField = int(popSize*(1-cleanSeed[year+1])/multiplicationRatio[0])

        cassavaPop = pd.DataFrame(np.zeros((popSize,len(colList))), index = np.arange(popSize), columns=colList)
        cassavaPop['yearPlanted'] = -1

        for indexStart in range(int(xdim)):
            cassavaPop['yPos'][int(ydim*indexStart):int(ydim*(indexStart+1))] = np.arange(0,ydim)
            cassavaPop['xPos'][int(ydim*indexStart):int(ydim*(indexStart+1))] = [indexStart]*int(ydim)
        
        if fieldType == 'replant': #this is the setup for farmer's fields
            cassavaPop['livePlants'] = 1
            cassavaPop['Whitefly'] = initialWhitefly

            #randomize infected stakes every year
            initialInfectedPlants = random.sample(range(int(popSize*cleanSeed[year]), popSize),int(initialInfection*popSize*(1-cleanSeed[year])))
            if len(sys.argv) > 3: #save field immages
                if sys.argv[3] == 'randomCleanSeed':
                    initialInfectedPlants = random.sample(range(popSize),int(initialInfection*popSize*(1-cleanSeed[year])))

            elif cleanSeed[year] != 0: #if there is clean seed
                cleanSeedPlants = range(int(popSize*(cleanSeed[year])))
                cassavaPop['yearPlanted'][cleanSeedPlants] = year
                if currentTime < pesticideDone: #add clean seed coating
                    cassavaPop['Whitefly'][cleanSeedPlants] = initialWhiteflyPesticide
                
                if year == 0 and initialInfection != 0: #this fixes a rounding to zero issue if there's one infected plant and clean seed
                    initialInfectedPlants = random.sample(range(int(popSize*cleanSeed[year]), popSize),max(1,int(initialInfection*popSize*(1-cleanSeed[year]))))

            cassavaPop['Infected'][initialInfectedPlants] = 1
            cassavaPop['SymptomDate'][initialInfectedPlants] = startTime
            cassavaPop['Proportion'][initialInfectedPlants] = 1
            cassavaPop['Symptoms'][initialInfectedPlants] = 0
            
        if fieldType == 'ratoon': #this is the setup for clean seed multiplication
            if year == 0:
                cassavaPop['livePlants'] = 1
                if replantFreq != 0:
                    indexStart = 0
                    planted = 1
                    numSections = int(int(xdim)/replantFreq)
                    for i in range(int(xdim)):
                        if i%numSections == 0 and planted > -replantFreq + 1: 
                            planted -= 1
                        cassavaPop['yearPlanted'][int(ydim*indexStart):int(ydim*(indexStart+1))] = planted
                        indexStart += 1
                        
                #initial infected stakes
                initialInfectedPlants = random.sample(range(popSize),int(initialInfection*popSize))
                
                cassavaPop['Infected'][initialInfectedPlants] = 1
                cassavaPop['SymptomDate'][initialInfectedPlants] = startTime
                cassavaPop['Proportion'][initialInfectedPlants] = 1
                cassavaPop['Symptoms'][initialInfectedPlants] = 0     
                    
            else:
                #keep field from previous growing season and make infected plants start fully infected
                cassavaPop = initialField
                
                #if plants are infected and alive:
                cassavaPop['Symptoms'][(cassavaPop['Infected'] == 1) & (cassavaPop['livePlants'] == 1)] = 1
                cassavaPop['Proportion'][(cassavaPop['Infected'] == 1) & (cassavaPop['livePlants'] == 1)] = 1
                cassavaPop['SymptomDate'][(cassavaPop['Infected'] == 1) & (cassavaPop['livePlants'] == 1)] = startTime

                #replant some chunk of the field here (assumes that the number of whitefly stays the same and that field is replanted with clean seed)
                newPlants = cassavaPop.loc[year - cassavaPop['yearPlanted'] == replantFreq].index

                cassavaPop['yearPlanted'][newPlants] = year
                cassavaPop['Infected'][newPlants] = 0
                cassavaPop['SymptomDate'][newPlants] = 0
                cassavaPop['Proportion'][newPlants] = 0
                cassavaPop['Symptoms'][newPlants] = 0
                cassavaPop['livePlants'][newPlants] = 1
                                                
                #update distanceMatrix to include all live plants, dead ones stay zeros
                aliveGrid = np.ix_(list(cassavaPop[cassavaPop['livePlants'] == 1].index), list(cassavaPop[cassavaPop['livePlants'] == 1].index))  
                distanceMatrix[aliveGrid] = distanceMatrixOriginal[aliveGrid]
                                    
                probList = np.zeros((popSize,popSize))        
                scale = np.linalg.norm(distanceMatrix, 1, axis = 1)
                for row in range(len(distanceMatrix)):
                    if scale[row] != 0:
                        distanceMatrix[row] /= scale[row]
                    probList[row] = np.cumsum(distanceMatrix[row])
            
            cassavaPop['Whitefly'][cassavaPop['livePlants'] == 1] = initialWhitefly
                    
            if currentTime < pesticideDone: #add clean seed coating to new plants
                cassavaPop['Whitefly'][cassavaPop['yearPlanted'] == year] = initialWhiteflyPesticide

            cassavaPop['Whitefly'] = cassavaPop['Whitefly'].astype('int64')

        #initialize lists
        infectedPlants = []
        roguingResults = []

        if currentTime > startTime:
            infectedPlants = [0.0]*int(currentTime-startTime)

        infectedPlants.append(sum(cassavaPop['Infected']))
                
        #simulation of growing season starts
        while currentTime < maxTime:            
            if len(cassavaPop[cassavaPop['livePlants'] == 1]) > 0: #immigrating whitefly
            
                #determine the number of immigrating whitefly
                nWhiteflyToday = math.floor(infectedWhiteflyIn)
                fractionalWhitefly = infectedWhiteflyIn - nWhiteflyToday
                if random.random() < fractionalWhitefly:
                    nWhiteflyToday += 1
                
                #select destination plants for the immigrating whitefly and whether the plant becomes infected
                for count in range(int(nWhiteflyToday)):    
                    immigrantPlant = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index)
                    if cassavaPop.iloc[immigrantPlant]['yearPlanted'] != year or currentTime > pesticideDone: #check that the immigrant plant doesn't have active pesticide
                        cassavaPop['Whitefly'][immigrantPlant] += 1
                        if cassavaPop['Infected'][immigrantPlant] == 0 and random.random() < pInfectingCassava:
                            cassavaPop['Infected'][immigrantPlant] = 1
                            cassavaPop['SymptomDate'][immigrantPlant] = currentTime + symptomDelay
                        
                        #to maintain a constant number of whitefly, choose a random whitefly to emigrate
                        plantIndex = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index)
                        while cassavaPop['Whitefly'][plantIndex] == 0: #this fixes an bug with neg whitefly on a plant
                            plantIndex = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index)
                        cassavaPop['Whitefly'][plantIndex] -= 1
                    
            #move whitefly within the field and determine if new plants become infected
            cassavaPop['Whitefly'] = cassavaPop['Whitefly'].astype('int64')
            whiteflyChange = np.zeros(popSize, dtype = 'int64')
            if len(cassavaPop[cassavaPop['livePlants'] == 1]) > 1: #if there's only one live plant, don't try to move whitefly
                fliesMoving = np.random.binomial(cassavaPop['Whitefly'],pLeavingPlant) #select how many flies are moving
                whiteflyChange -= fliesMoving                               
                
                infectedFlies = np.random.binomial(fliesMoving,(pInfectingWhitefly * cassavaPop['Proportion']*pInfectingCassava))                
                uninfectedFlies = fliesMoving - infectedFlies
                
                alternativePlants = cassavaPop[(cassavaPop['livePlants'] == 1) & (cassavaPop['yearPlanted'] != year)].index
                pesticidePlants = cassavaPop[cassavaPop['yearPlanted'] == year].index #plants that had pesticide at some point this growth season

                for plant in range(popSize): #move infected flies and check for new infections
                    for iFly in range(infectedFlies[plant]):
                        i = find_gt(probList[plant], random.random())
                        
                        if i in pesticidePlants and currentTime < pesticideDone and len(alternativePlants) > 0: #if pesticide kills the whitefly, add a new one to a random plant without pesticide
                            whiteflyChange[np.random.choice(alternativePlants)] += 1                            
                        
                        else:
                            whiteflyChange[i] += 1
                            if cassavaPop['Infected'][i] == 0: #if the destination plant isn't infected
                                cassavaPop['Infected'][i] = 1
                                cassavaPop['SymptomDate'][i] = currentTime + symptomDelay

                    for uFly in range(uninfectedFlies[plant]): #move uninfected flies
                        i = find_gt(probList[plant], random.random())
                        
                        if i in pesticidePlants and currentTime < pesticideDone and len(alternativePlants) > 0: #if pesticide kills whitefly, add a new one to a random plant w/o pesticide
                            whiteflyChange[np.random.choice(alternativePlants)] += 1
                        
                        else:
                            whiteflyChange[i] += 1

            cassavaPop['Whitefly'] = cassavaPop['Whitefly'] + whiteflyChange #update fly counts on each plant
            
            #increase infection proportion
            proportionIndexList = cassavaPop[(cassavaPop['Infected'] == 1) & (cassavaPop['Proportion'] < 1)]['Proportion'].index
            cassavaPop.loc[proportionIndexList, 'Proportion'] += 1.0/fullInfection #if partially infected, become more infected
            cassavaPop['Proportion'][np.where(cassavaPop['Proportion'] > 1)[0]] = 1
                                
            #increase symptom proportion 
            indexList = cassavaPop[(cassavaPop['Infected'] == 1) & (cassavaPop['SymptomDate'] < currentTime) & (cassavaPop['Symptoms'] < maxSymptoms)]['Symptoms'].index
            cassavaPop.loc[indexList, 'Symptoms'] += 1.0/fullSymptoms * maxSymptoms
            cassavaPop['Symptoms'][np.where(cassavaPop['Symptoms'] > maxSymptoms)[0]] = maxSymptoms

            #after pesticide is done, increase whitefly on all plants
            if currentTime == pesticideDone:
                cassavaPop['Whitefly'][cassavaPop['Infected'] == 1][cassavaPop['yearPlanted'] == year] += (initialWhitefly - initialWhiteflyPesticide)
                
            #do roguing
            if currentTime in roguingTimes:
                out, probList = roguing(cassavaPop, probList)
                roguingResults.append(out)
                
            #survey plants
            if currentTime in surveyTimes:
                for plants in plantsSurveyed:
                    for accuracy in surveyAccuracy:
                        allSurveys[str(accuracy)+'-'+str(plants)+'-'+str(year)+'-'+str(currentTime)] = surveyPlants(currentTime, cassavaPop, plants, accuracy)


            #do preferential selection
            if currentTime in selectionTimes and len(cassavaPop[cassavaPop['livePlants'] == 1]) >= plantsToReplantField: #if there are enough plants to replant the field
                if cleanSeed[year] != 0: #start by selecting plants from the part of the field planted with clean seed
                    plantsForNextSeason = []

                    minFullyInfected = plantsToReplantField - len(cassavaPop[(cassavaPop['livePlants'] == 1) & (cassavaPop['Symptoms'] < maxSymptoms)])

                    if minFullyInfected > 0: #if some fully infected plants need to be included add them here (fixes an infinite loop issue if roguingAccuracy == 1)
                        fullyInfectedPlants = cassavaPop[(cassavaPop['livePlants'] == 1) & (cassavaPop['Symptoms'] >= maxSymptoms)].index
                        plantsForNextSeason = fullyInfectedPlants[:int(minFullyInfected)].values.tolist()
                    
                    counter = 0    
                    while len(plantsForNextSeason) < plantsToReplantField: #keep looping through the plants in the field until enough are selected
                        item = cassavaPop[cassavaPop['livePlants'] == 1].index[counter]
                        if item not in plantsForNextSeason: 
                            if (1 - cassavaPop.iloc[item]["Symptoms"]*roguingAccuracy) > random.random():
                                plantsForNextSeason.append(item)
                        counter += 1
                        if counter == len(cassavaPop[cassavaPop['livePlants'] == 1]):
                            counter = 0
                                
                    plantsForNextSeason = np.asarray(plantsForNextSeason)
                else: 
                    minFullyInfected = plantsToReplantField - len(cassavaPop[(cassavaPop['livePlants'] == 1) & (cassavaPop['Symptoms'] < maxSymptoms)])
                    
                    if minFullyInfected > 0 and maxSymptoms*roguingAccuracy >= 1: #if some fully infected plants need to be included add them here (fixes an infinite loop issue if roguingAccuracy == 1)
                        fullyInfectedPlants = cassavaPop[(cassavaPop['livePlants'] == 1) & (cassavaPop['Symptoms'] >= maxSymptoms)].index
                        plantsForNextSeason = fullyInfectedPlants[:int(minFullyInfected)].values
                                          
                        if len(plantsForNextSeason) < plantsToReplantField: #fixes numpy runtime error 
                            col = (1 - cassavaPop[cassavaPop['livePlants'] == 1]["Symptoms"]*roguingAccuracy).abs() #probability of choosing a plant                    
                            plantsRandom = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index, replace = False, size = plantsToReplantField-minFullyInfected, p = col/sum(col))
                            plantsForNextSeason = np.concatenate([plantsForNextSeason,plantsRandom])
                    
                    else:
                        col = (1 - cassavaPop[cassavaPop['livePlants'] == 1]["Symptoms"]*roguingAccuracy).abs() #probability of choosing a plant                    
                        plantsForNextSeason = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index, replace = False, size = plantsToReplantField, p = col/sum(col))

            if len(sys.argv) > 3: #save images of the field each day of the growing season
                if sys.argv[3] == 'video': #using extent and aspect takes into account the different x and y spacing    
                    imageNumber = (year*(maxTime-startTime))+(currentTime-startTime)
                    plt.cla()
                    plotty = np.copy(cassavaPop['Infected']).reshape((int(xdim),int(ydim)))
                    plt.title('Year '+str(int(year)), fontsize = 20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(np.rot90(plotty), cmap='viridis', interpolation = 'None', vmax = 1, vmin = 0, extent=[0,xdim*spacingX,0,ydim*spacingY], aspect='equal')
                    plt.savefig("%04d" % imageNumber + 'json'+argv+'.png')    
                        
            currentTime += 1

            infectedPlants.append(sum(cassavaPop['Infected']))
        
        #choose plants at the end of the season to use for replanting the following year if there's no preferential selection
        if len(selectionTimes) == 0 and len(cassavaPop[cassavaPop['livePlants'] == 1]) >= plantsToReplantField and fieldType == 'replant':
            if cleanSeed[year] == 0: #choose randomly if no clean seed
                plantsForNextSeason = np.random.choice(cassavaPop[cassavaPop['livePlants'] == 1].index, replace = False, size = plantsToReplantField)
         
            else: #choose as many plants from the clean seed block as possible
                plantListnp = np.asarray(cassavaPop[cassavaPop['livePlants'] == 1].index)
                if len(plantListnp[plantListnp < popSize*cleanSeed[year]]):
                    plantsForNextSeason = np.random.choice(plantListnp[plantListnp < popSize*cleanSeed[year]], replace = False, size = plantsToReplantField)

                else: #take all clean seed plants plus others
                    clean = plantListnp[plantListnp < popSize*cleanSeed[year]]
                    other = np.random.choice(plantListnp[plantListnp >= popSize*cleanSeed[year]], replace = False, size = plantsToReplantField-len(clean))
                    plantsForNextSeason = np.concatenate((clean,other))
    
        #if the field is dead or there's ratooning
        if len(cassavaPop[cassavaPop['livePlants'] == 1]) < plantsToReplantField or fieldType == 'ratoon': 
            plantsForNextSeason = []

        return infectedPlants, roguingResults, cassavaPop, probList, plantsForNextSeason
   
    #parameters constant across all simulations
    maxTime = 300 #Duration (in days) of the growing season from planting to harvest
    symptomDelay = 30 #Duration (in days) from infection to first onset of visible symptoms
    pInfectingWhitefly = 1.0 #Probability of a whitefly carrying infection when leaving an infected plant (left as 1.0 w.l.o.g. due to degeneracy with pLeavingPlant and for increased computational efficiency)
    spacingX = 1 #Distance (in metres) between cassava plants within rows 
    spacingY = 1.5 #Distance (in metres) between rows of cassava plants 
    fullSymptoms = 60.0 #number of days AFTER the symptomDelay that it takes a simulated cassava plant to become 100% symptomatic 
    startTime = 30.0 #Time (in days) after planting that an infected cuttting can start to infect other plants
    year = 0 #current simulation year, starting from year zero
    colList = ['Infected','SymptomDate','Proportion','Whitefly','xPos','yPos', 'Symptoms', 'yearPlanted', 'livePlants'] #enumeration of the properties of a simulated cassava plant
    #Infected: boolean representing current infection status of the cassava plant
    #SymptonDate: Simulation day on at which symptoms will first appear
    #Proportion: Current proportional (0.0 - 1.0) of systemicity of infection. Used to modulate the probability of whitefly transmitting infection 
    #Whitefly: Integer count of current whitefly on plant
    #xPos, yPos: spatial coordinates of simulated cassava plant within simulated cassava field
    #Symptoms: Current proportional (0.0 - 1.0) of systemicity of infection. Used to modulate the probability of surveys successfully identifying a plant as infected
    #yearPlanted: simulation year in which this simulated cassava plant was planted
    #livePlants: boolean representing if the cassava plant is currently alive

    #parameters from input file
    argv = str(argv)
    with open(argv+'.json') as f:
        data = json.load(f)
        
    pLeavingPlant = data["pLeavingPlant"] #Probability of a whitefly leaving a cassava plant each simulation day
    pInfectingCassava = data["pInfectingCassava"]/pInfectingWhitefly #probability of an infected whitefly that lands on a susceptible cassava plant causing an infection in that plant
    kernelScale = data["kernelScale"] #scale parameter for exponential dispersal kernel
    infectedWhiteflyIn = data["infectedWhiteflyIn"] #Number of whitefly immigrating to the field each simulation day
    startingInfection = data["startingInfection"]  #Proportion (0.0 - 1.0) of the field to begin as infected -normal limit for inspected stakes is 10%
    maxSymptoms = data["maxSymptoms"] #maximum symptom expression level (on 0.0 to 1.0 scale) reached in simulated cassava cultivar. Modulates successful survey identification probability. Typically 1.0
    endYear = data["endYear"] #Simulation year in which to end the simulation; duration of simulation in simulation years
    initialWhitefly = data["initialWhitefly"] #initial count of simulated whitefly on simulated cassava plants that have not had pesticide applied
    roguingAccuracy = data["roguingAccuracy"] #Base probability (0.0 - 1.0) of roguing successfully identifying and removing a fully symptomatic plant
    roguingTimes = data["roguingTimes"] #array of simulation dates (in days) on which to conduct simulated roguing sweeps each simulation year
    multiplicationRatio = data["multiplicationRatio"] #number of cuttings produced from one cassava plant used when replanting the field from the previous year
    startTimeZero = data["startTimeZero"] #Simulation day on which to initialise the field in the first year - used to specify the initial introduction date to simulate scenarios in which the field is not planted with infected material
    popSize = data["popSize"] #Number of simulated cassava plants in the cassava field
    fullInfection = data["fullInfection"] #Number of days after infection for infection in a plant to become fully systemic
    
    #management specific parameters
    cleanSeed = data["cleanSeed"] #array specifying the proportion of the field to be supplied with clean planting material for each year of the simulation
    selectionTimes = data["selectionTimes"] #Array of simulation dates (in days) on which to conduct a simulated survey and flag plants as clean for use in next years replanting. Used each simulation year
    fieldType = data["fieldType"] #Specifies whether the field is used managed for the production of root matter and replanted each year, or for production of cuttings and ratooned each year
    replantFreq = data["replantFreq"] #If a field is ratooned, the field is replanted every replantFreq simulation years
    pesticideDone = data["pesticideDone"] #Simulation date (in days post planting) on which an application of pesticide applied at the start of the season will expire
    initialWhiteflyPesticide = data["initialWhiteflyPesticide"] #Initial count of simulated whitefly present on simulated cassava plants which have had pesticide applied

	#survey specific parameters
    surveyAccuracy = data["surveyAccuracy"] #Probability (0.0 - 1.0) of a surveyor successfully detecting infection in a fully symptomatic plant
    plantsSurveyed = data["plantsSurveyed"] #Number of plants to be surveyed within the field in a simulated survey sweep
    surveyTimes = data["surveyTimes"] #Array of dates (in days) on which to conduct survey sweeps each simulation year

   
    #Validation of data read from input files:
    assert fieldType == 'replant' or fieldType == 'ratoon'
    assert len(cleanSeed) >= endYear
    assert type(pesticideDone) != list
    
    #calculate field shape and assign x and y pos to each plant
    xdim = 1.0
    ydim = 1.0
    factorList = prime_factors(popSize)
    for i in range(len(factorList)):
        if xdim < ydim:
                xdim *= factorList[-i]
        else:
            ydim *= factorList[-i]
    
    #calculate distances between each plant and the probability of a whitefly moving between any pair of plants
    distX = np.zeros(popSize)
    distY = np.zeros(popSize)
    
    indexStart = 0
    for i in range(int(xdim)):
        distY[int(ydim*indexStart):int(ydim*(indexStart+1))] = np.arange(0,ydim)
        distX[int(ydim*indexStart):int(ydim*(indexStart+1))] = [indexStart]*int(ydim)
        indexStart += 1
    
    distanceMatrixOriginal = np.zeros((popSize,popSize))
    for row in range(popSize):
        rowValX = distX[row]
        rowValY = distY[row]
        for col in range(popSize):
            if row != col: #probability of staying at plant is zero if already chosen to leave
                distanceMatrixOriginal[row][col] = np.e**(-kernelScale*math.sqrt(((rowValX-distX[col])*spacingX)**2 + ((rowValY - distY[col])*spacingY)**2))

    #convert the distances between plants into the probability of a whitefly traveling between any pair of plants
    distanceMatrix = np.copy(distanceMatrixOriginal) #reset these to the original                
    scale = np.linalg.norm(distanceMatrix, 1, axis = 1)
    for row in range(len(distanceMatrix)):
        distanceMatrix[row] /= scale[row]
    probList = np.zeros((popSize,popSize))
    for row in range(len(distanceMatrix)):
        probList[row] = np.cumsum(distanceMatrix[row])
    
    #set up dataframes to save simulation outputs
    colList1 = ['Time']
    for col in range(endYear):
            colList1.append(col)
    overalInfection = pd.DataFrame(np.zeros((int(maxTime - startTime+1),len(colList1))), index = np.arange(int(maxTime - startTime+1)), columns=colList1)
    overalInfection['Time'] = np.arange(startTime, maxTime+1)
    
    colList2 = ['Xpos','Ypos']
    for col in range(endYear):
        colList2.append(col)
    overalInfectionSpatial = pd.DataFrame(np.zeros((popSize,len(colList2))), index = np.arange(popSize), columns=colList2)
    overalInfectionSpatial['Xpos'] = distX
    overalInfectionSpatial['Ypos'] = distY

    colList3 = []
    for col in range(endYear):
        colList3.append(col)
    overalRoguing = pd.DataFrame(np.zeros((len(roguingTimes),endYear)), index = roguingTimes, columns=colList3)

    if len(surveyTimes) > 0:
        colList4 = [str(accuracy)+'-'+str(plants)+'-'+str(year)+'-'+str(time) for accuracy in surveyAccuracy for plants in plantsSurveyed for year in range(endYear) for time in surveyTimes]
        allSurveys = pd.DataFrame(np.zeros((1,len(colList4))), index = np.arange(1), columns=colList4)

    #start running the simulation here
    while year < endYear:        
        if year == 0:
            infection, currentRoguing, fieldStatus, probList, startingPlants = growingSeason(startTimeZero,probList, initialInfection = startingInfection)
 
            overalInfection[year] = infection
            overalInfectionSpatial[year] = fieldStatus['Infected']
            overalRoguing[year] = currentRoguing

            year += 1
            
        else:
            if fieldType == 'ratoon':
                #note: distanceMatrix and problist are updated at begining of year after new plants are added
                
                #start simulating the next growing season
                infection, currentRoguing, fieldStatus, probList, startingPlants = growingSeason(startTime, probList, initialField = fieldStatus)
    
    			#saving outputs
                overalInfection[year] = infection
                overalInfectionSpatial[year] = fieldStatus['Infected']
                overalRoguing[year] = currentRoguing

                year += 1

            if fieldType == 'replant':
                if len(startingPlants) != 0: #if there are enough plants to repopulate field
                    #reset distanceMatrix, probList to original values b/c entire field is replanted
                    distanceMatrix = np.copy(distanceMatrixOriginal)
                    scale = np.linalg.norm(distanceMatrix, 1, axis = 1)
                    for row in range(len(distanceMatrix)):
                        distanceMatrix[row] /= scale[row]
                    probList = np.zeros((popSize,popSize))
                    for row in range(len(distanceMatrix)):
                        probList[row] = np.cumsum(distanceMatrix[row])

                    #start simulating the next growing season
                    startingInfection = fieldStatus.loc[startingPlants]['Proportion'].mean()                    
                    infection, currentRoguing, fieldStatus, probList, startingPlants = growingSeason(startTime, probList,initialInfection = startingInfection)
        
                    #saving outputs       
                    overalInfection[year] = infection
                    overalInfectionSpatial[year] = fieldStatus['Infected']
                    overalRoguing[year] = currentRoguing
                
                    year += 1    
                                        
                else: #if there aren't enought plants to replant the field, record it and move on to the next input file
                    overalInfection.to_csv('DEAD_overalInfectionDead'+argv+'.csv', index=False)
                    overalInfectionSpatial.to_csv('DEAD_overalInfectionSpatial'+argv+'.csv', index=False)
                    overalRoguing.to_csv('DEAD_overalRoguing'+argv+'.csv', index=False)
        
                    with open('deadField'+argv+'.txt', 'w') as f:
                        f.write("This field ran out of plants")          

                    return

    overalInfection.to_csv('overalInfection'+argv+'.csv', index=False)
    overalInfectionSpatial.to_csv('overalInfectionSpatial'+argv+'.csv', index=False)
    overalRoguing.to_csv('overalRoguing'+argv+'.csv', index=False)
    if len(surveyTimes) > 0:
         allSurveys.to_csv('allSurveys'+argv+'.csv', index=False)

numToRun = sys.argv[1]
batchNum = sys.argv[2]

print len(sys.argv), sys.argv[:]

start = int(batchNum)*int(numToRun)
end = start + int(numToRun)
#sys.stdout = Unbuffered(sys.stdout)

#Run simulations:
for i in range(start, end):
    if not os.path.isfile('overalInfection'+str(i)+'.csv') and not os.path.isfile('deadField'+str(i)+'.txt'):
        print 'starting simulation', i
        simulation(i)
