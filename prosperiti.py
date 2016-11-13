import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

__author__ = 'Joe Cursons'
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This python script accompanies the book chapter:
#  Proteome Bioinformatics, Methods in Molecular Biology. Edited by Shivakumar Keerthikumar and Suresh Mathivanan.
#  Chapter 15: Permutation testing to examine the significance of network features in protein-protein interaction
#  networks.
#   Written by Joe Cursons & Melissa Davis.
#   © Springer Science+Business Media LLC 2017.
#   ISBN: 978-1-4939-6738-4
#   DOI: 10.1007/978-1-4939-6740-7_15
#
# It has been designed to provide a worked example of how permutation testing can be applied to understand the
#   significance of network features such as connectivity when examining proteomics data.
#
# Please note that this code may be updated over time as new functionality is added. A version of this script which
#  has been validated as re-producing the figures from the textbook is available from:
#   http://dx.doi.org/10.5281/zenodo.166341
#
# This script contains several functions to extract and process the input data:
#
#
# These functions are executed over lines
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# In this example, we use a published phospho-tyrosine enriched quantitative MS/MS data set:
#   Hochgrafe F, Zhang L, O'Toole SA, Browne BC, Pinese M, Porta Cubas A, Lehrbach GM, Croucher DR, Rickwood D,
#       Boulghourjian A, Shearer R, Nair R, Swarbrick A, Faratian D, Mullen P, Harrison DJ, Biankin AV, Sutherland RL,
#       Raftery MJ, Daly RJ.
#   Tyrosine phosphorylation profiling reveals the signaling network characteristics of Basal breast cancer cells.
#   Cancer Research. 2010 Nov 15; 70(22): 9391-401.
#
#   http://dx.doi.org/10.1158/0008-5472.CAN-10-0911
#
# These data (Table S3) are directly available from:
#   http://cancerres.aacrjournals.org/content/70/22/9391/suppl/DC1
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# In this example, we use a published protein-protein interact data set:
#   Cowley MJ, Pinese M, Kassahn KS, Waddell N, Pearson JV, Grimmond SM, Biankin AV, Hautaniemi S, Wu J.
#   PINA v2.0: mining interactome modules.
#   Nucleic acids research. 2012;40(Database issue):D862-5.
#
#   http://dx.doi.org/10.1093/nar/gkr967
#
# These data are directly available from:
#   http://cbg.garvan.unsw.edu.au/pina/download/Homo%20sapiens-20140521.tsv
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This python script was written by:
#       Joe Cursons, University of Melbourne Systems Biology Laboratory - joseph.cursons@unimelb.edu.au
#       Melissa Davis, Walter and Eliza Hall Institute - melissa.davis@unimelb.edu.au
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Extract:

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that reads in the Hochgrafe data and exports lists of proteins detected (UniProt IDs) for the specified
    #   cell lines
    # inputs:
    #   structInHochgrafeData
    #       'CellLines' - a list of strings referencing the cell lines from the Hochgrafe data set
    #       'UniProt' - a list of UniProt IDs for detected proteins within the Hockgrafe data
    #       'arrayProtAbund' - a 2D array (numUniProtIDs = numRows; numCellLines = numCols) containing peptide/protein
    #                           abundance data from Hochgrafe et al
    #   strInCellLinesOfInt
    #       string containing the cell line of interest from the Hochgrafe data
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def hochgrafe_lists(structInHochgrafeData, strInCellLine):

        # determine the index of this cell line within the Hochgrafe data
        listCellLines = structInHochgrafeData['CellLines']
        numCellLineIndex = listCellLines.index(strInCellLine)

        # the Hochgrafe data contain some 'multiple entry' proteins due to peptides with identity across multiple
        #  proteins, these 'shared peptide sequences' often map to proteins which form large signalling complexes and
        #  thus these entries are excluded to prevent excess influence upon network statistics (when included multiple
        #  times)
        listUniProtEntries = structInHochgrafeData['UniProt']
        arrayListWithMultipleEntryFlag = np.zeros(len(listUniProtEntries),dtype=np.bool)
        for iEntry in range(len(listUniProtEntries)):
            if '/' in listUniProtEntries[iEntry]:
                arrayListWithMultipleEntryFlag[iEntry] = True
        arrayListWithSingleEntryIndices = np.where(arrayListWithMultipleEntryFlag == False)[0]

        # the background list is simply the full list without any 'multiple entry' components
        listBackground = [structInHochgrafeData['UniProt'][i] for i in arrayListWithSingleEntryIndices]

        # the cell-line specific data should contain proteins detected for that cell line
        arrayCellLineAbundData = structInHochgrafeData['arrayProtAbund'][:,numCellLineIndex]
        arrayProteinDetectedFlag = arrayCellLineAbundData > 0

        # take the intersection of the unique proteins and detected proteins for this cell line
        arrayOutputRows = np.where((arrayProteinDetectedFlag == True) & (arrayListWithMultipleEntryFlag == False))[0]
        arrayCellLineUniProtRows = [structInHochgrafeData['UniProt'][i] for i in arrayOutputRows]

        return {'UniProtBackground':listBackground, 
                'UniProtListByCondition':arrayCellLineUniProtRows, 
                'Condition':strInCellLine}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that specifically loads the protein data (stab_3.xsl) from Hochgrafe et. al (2010) and exports all
    #   listed fields
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def hochgrafe_supp_table_3(strInFolderPath, flagPerformExtraction):
        # strInFilePointer: absolute folder path for stab_3.xls
        # flagPerformExtraction: Boolean variable specifying whether or not the data file should be re-extracted; or
        #   whether a temporary saved file can be used to reduce run-time

        # set the input/output file names
        strDataFile = 'stab_3.xls'
        strOutputSaveFile = 'processedProteinData'

        if flagPerformExtraction:
            print('Attempting to extract protein-level phospho-tyrosine enriched MS data from Hochgrafe et al. (2010)')
            # load the file into a dataframe using pandas
            dfProteinInfo = pd.read_excel(os.path.join(strInFolderPath, strDataFile),
                                                  sheetname='Total phosphorylation')

            # the header row (will cell lines) is on row 7 -> 6 with an index-0 array
            listCellLines = dfProteinInfo.iloc[6,:].values.tolist()
            # and cell lines are stored in position 5-> (index-0 array)
            listCellLines = listCellLines[5:]
            numCellLines = len(listCellLines)

            # the protein HGNC list is contained in column 1, from row 7 onwards (index-0 arrays)
            listTargProtHGNC = dfProteinInfo.iloc[:,1].values.tolist()
            arrayProteinHGNCs = listTargProtHGNC[7:]
            numTargets = len(arrayProteinHGNCs)

            # and UniProt IDs are contained in column 0, from row 7 onwards (index-0 arrays)
            listTargProtUniProt = dfProteinInfo.iloc[:,3].values.tolist()
            arrayProteinUniProtIDs = listTargProtUniProt[7:]

            # create a numpy array to store the numerical data
            arrayProtAbund = np.zeros((numTargets, numCellLines), dtype=np.float32)
            for iProt in range(numTargets):
                numRow = iProt + 7
                arrayDataRow = dfProteinInfo.iloc[numRow,:].values
                for iCol in range(numCellLines):
                    arrayProtAbund[iProt,iCol] = np.float32(arrayDataRow[iCol+5])

            # save the output arrays using numpy (as this is quicker than reading csv files back in)
            np.savez(os.path.join(strInFolderPath, strOutputSaveFile),
                     arrayCellLines=listCellLines,
                     arrayProteinHGNCs=arrayProteinHGNCs,
                     arrayProteinUniProtIDs=arrayProteinUniProtIDs,
                     arrayProtAbund=arrayProtAbund)
        else:
            if os.path.exists(os.path.join(strInFolderPath, (strOutputSaveFile + '.npz'))):
                # load the data from the specified files
                print('Loading the processed protein data data from ' +
                      os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')))

                npzfile = np.load(os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')))

                listCellLines = npzfile['arrayCellLines']
                arrayProteinHGNCs = npzfile['arrayProteinHGNCs']
                arrayProteinUniProtIDs = npzfile['arrayProteinUniProtIDs']
                arrayProtAbund = npzfile['arrayProtAbund']

            else:
                print('Cannot load the processed protein data, ' +
                      os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')) +
                      ' does not exist, change flagPerformExtraction')

        return {'HGNC': arrayProteinHGNCs,
                'UniProt':arrayProteinUniProtIDs,
                'CellLines':listCellLines,
                'arrayProtAbund':arrayProtAbund}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that specifically loads the protein-protein interaction data from PINA v2.0 (through the MI-TAB tsv)
    #   and exports all proteins (UniProt ID and corresponding protein name), together with a connectivity matrix
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def pina2_mitab(strInFolderPath, flagPerformExtraction):
        # strInFilePointer: absolute folder path for Homo sapiens-20140521.tsv
        # flagPerformExtraction: Boolean variable specifying whether or not the data file should be re-extracted; or
        #   whether a temporary saved file can be used to reduce run-time

        # set the input/output file names
        strDataFile = 'Homo sapiens-20140521.tsv'
        strOutputSaveFile = 'processedPINA2Data'

        if flagPerformExtraction:
            print('Extracting PINA (v.2.0) database into arrays appropriate for computational analysis')
            # load the file and determine its length
            dataFramePINA2 = pd.read_table(os.path.join(strInFolderPath, strDataFile))

            dataFramePINA2UniProts = dataFramePINA2.loc[:,'ID(s) interactor A':'ID(s) interactor B']
            arrayUniqueUniProtIDs = pd.unique(dataFramePINA2UniProts.values.ravel())
            numUniqueProts = len(arrayUniqueUniProtIDs)
            listUniqueUniProtIDs = list(arrayUniqueUniProtIDs)

            # strip out the 'uniprotkb:' string preceding every entry to get the output list
            listOutputUniProtIDs = []
            listOutputProteinHGNCs = []
            for stringProtID in arrayUniqueUniProtIDs:
                arraySplitEntry = stringProtID.split(':')
                listOutputUniProtIDs.append(arraySplitEntry[1])

                strHGNC = 'failed_map'
                arrayRowIndicesIntA = np.where(dataFramePINA2['ID(s) interactor A'].values == stringProtID)[0]

                if len(arrayRowIndicesIntA) > 0:
                    numFirstRow = arrayRowIndicesIntA[0]
                    strAltName = dataFramePINA2['Alt. ID(s) interactor A'].iloc[numFirstRow]
                    arraySplitAlt = strAltName.split(':')
                    strNameAndExtra = arraySplitAlt[1]
                    if strNameAndExtra[-11:] == '(gene name)':
                        strHGNC = strNameAndExtra[0:-11]

                else:
                    arrayRowIndicesIntB = np.where(dataFramePINA2['ID(s) interactor B'].values == stringProtID)[0]
                    numFirstRow = arrayRowIndicesIntB[0]
                    strAltName = dataFramePINA2['Alt. ID(s) interactor B'].iloc[numFirstRow]
                    arraySplitAlt = strAltName.split(':')
                    strNameAndExtra = arraySplitAlt[1]
                    if strNameAndExtra[-11:] == '(gene name)':
                        strHGNC = strNameAndExtra[0:-11]

                listOutputProteinHGNCs.append(strHGNC)

            arrayInteractionNetwork = np.zeros((numUniqueProts,numUniqueProts), dtype=np.bool_)
            for iRow in range(len(dataFramePINA2UniProts)):
                stringProtA = dataFramePINA2['ID(s) interactor A'][iRow]
                stringProtB = dataFramePINA2['ID(s) interactor B'][iRow]
                numRow = listUniqueUniProtIDs.index(stringProtA)
                numCol = listUniqueUniProtIDs.index(stringProtB)
                arrayInteractionNetwork[numRow, numCol] = True

            # save the output arrays using numpy (as this is quicker than reading csv files back in)
            np.savez(os.path.join(strInFolderPath, strOutputSaveFile),
                     arrayInteractionNetwork=arrayInteractionNetwork,
                     arrayOutputProteinHGNCs=listOutputProteinHGNCs,
                     arrayOutputUniProtIDs=listOutputUniProtIDs)
        else:
            if os.path.exists(os.path.join(strInFolderPath, (strOutputSaveFile + '.npz'))):
                # load the data from the specified files
                print('Loading the processed PINA2 data data from ' + os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')))

                npzFile = np.load(os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')))

                arrayInteractionNetwork = npzFile['arrayInteractionNetwork']
                listOutputProteinHGNCs = npzFile['arrayOutputProteinHGNCs']
                listOutputUniProtIDs = npzFile['arrayOutputUniProtIDs']

            else:
                print('Cannot load the processed protein data, ' +
                      os.path.join(strInFolderPath, (strOutputSaveFile + '.npz')) +
                      ' does not exist, change flagPerformExtraction')

        return {'HGNC':listOutputProteinHGNCs, 'UniProt':listOutputUniProtIDs, 'arrayIntNetwork':arrayInteractionNetwork}

class Build:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that specifically loads the protein-protein interaction data from PINA v2.0 (through the MI-TAB tsv)
    #   and exports all proteins (UniProt ID and corresponding protein name), together with a connectivity matrix
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def ppi_graph(structInPINANetwork, arrayProteinsInNetwork):

        # create the output network
        graphOutputNetwork = nx.Graph()

        # populate the output network with the desired proteins
        graphOutputNetwork.add_nodes_from(arrayProteinsInNetwork)

        # populate the output network with edges/relationships between the proteins
        for stringProtOne in arrayProteinsInNetwork:
            if stringProtOne in structInPINANetwork['UniProt']:
                numProtOneIndex = list(structInPINANetwork['UniProt']).index(stringProtOne)
                arrayInteractionPartnerFlag = structInPINANetwork['arrayIntNetwork'][numProtOneIndex,:]
                arrayInteractionPartnerIndices = np.where(arrayInteractionPartnerFlag)
                for numProtTwoIndex in arrayInteractionPartnerIndices[0]:
                    stringProtTwo = structInPINANetwork['UniProt'][numProtTwoIndex]
                    if stringProtTwo in arrayProteinsInNetwork:
                        graphOutputNetwork.add_edge(stringProtOne,stringProtTwo)
            else:
                # assume that this protein has no known PPIs, move on to the next protein in the list
                print('warning: ' + stringProtOne + ' can not be found within the protein-protein interaction data')

        return graphOutputNetwork

class Test:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def network_features(structInPINANetwork, arrayProteinList, numPermTests):

        # extract the corresponding network from the PINA2 data into a NetworkX graph
        graphNetwork = Build.ppi_graph(structInPINANetwork, arrayProteinList)
        #
        graphNetworkConnected = max(nx.connected_component_subgraphs(graphNetwork), key=len)
        numAvgClustering = nx.average_clustering(graphNetwork)
        numDiameter = nx.diameter(graphNetworkConnected)
        numConnectedNodes = nx.number_of_nodes(graphNetworkConnected)

        numNetworkNodes = len(arrayProteinList)

        arrayRandNetworkAvgClustering = np.zeros(numPermTests,dtype=np.float_)
        arrayRandNetworkDiameter = np.zeros(numPermTests,dtype=np.int32)
        arrayRandNetworkConnectedNodes = np.zeros(numPermTests,dtype=np.int32)

        for iPermTest in range(numPermTests):
            arrayRandUniProtIDs = np.random.choice(structInPINANetwork['UniProt'], numNetworkNodes)
            graphRandNetworkOfSameSize = Build.ppi_graph(structInPINANetwork, arrayRandUniProtIDs)
            graphRandNetworkOfSameSizeConnected = max(nx.connected_component_subgraphs(graphRandNetworkOfSameSize), key=len)

            numRandNetworkAvgClustering = nx.average_clustering(graphRandNetworkOfSameSize)
            numRandNetworkDiameter = nx.diameter(graphRandNetworkOfSameSizeConnected)
            numRandNetworkConnectedNodes = nx.number_of_nodes(graphRandNetworkOfSameSizeConnected)

            arrayRandNetworkAvgClustering[iPermTest] = numRandNetworkAvgClustering
            arrayRandNetworkDiameter[iPermTest] = numRandNetworkDiameter
            arrayRandNetworkConnectedNodes[iPermTest] = numRandNetworkConnectedNodes

            print('permutation test ' + str(iPermTest) + ' of ' + str(numPermTests) + ' completed')

        return {'arrayRandNetworkAvgClustering':arrayRandNetworkAvgClustering,
                'arrayRandNetworkDiameter':arrayRandNetworkDiameter,
                'arrayRandNetworkConnectedNodes':arrayRandNetworkConnectedNodes,
                'numAvgClustering':numAvgClustering,
                'numDiameter':numDiameter,
                'numConnectedNodes':numConnectedNodes}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # function that
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def edge_correlation(structInPINANetwork, structProteinData, numPermTests, flagSkipKnownPPIs):


        # the Hochgrafe data contain some 'multiple entry' proteins due to peptides with identity across multiple
        #  proteins, these 'shared peptide sequences' often map to proteins which form large signalling complexes and
        #  thus these entries are excluded to prevent excess influence upon network statistics (when included multiple
        #  times)
        listUniProtEntries = structProteinData['UniProt']
        arrayListWithMultipleEntryFlag = np.zeros(len(listUniProtEntries),dtype=np.bool)
        for iEntry in range(len(listUniProtEntries)):
            if '/' in listUniProtEntries[iEntry]:
                arrayListWithMultipleEntryFlag[iEntry] = True
        arrayListWithSingleEntryIndices = np.where(arrayListWithMultipleEntryFlag == False)[0]

        # the background list is simply the full list without any 'multiple entry' components
        listBackground = [structProteinData['UniProt'][i] for i in arrayListWithSingleEntryIndices]

        arrayAllProteinData = structProteinData['arrayProtAbund']
        arrayProteinData = np.zeros((np.size(structProteinData['arrayProtAbund'],0),
                                     np.size(structProteinData['arrayProtAbund'],1)),
                                    dtype=np.float_)
        for iRow in range(len(arrayListWithSingleEntryIndices)):
            arrayProteinData[iRow,:] = arrayAllProteinData[arrayListWithSingleEntryIndices[iRow],:]

        # extract the corresponding network from the PINA2 data into a NetworkX graph
        graphNetwork = Build.ppi_graph(structInPINANetwork, listBackground)

        arrayNetworkEdges = graphNetwork.edges()
        arrayEdgeCorr = np.zeros([graphNetwork.number_of_edges(),1], dtype=np.float_)
        for iEdge in range(graphNetwork.number_of_edges()):
            arrayEdge = arrayNetworkEdges[iEdge]
            stringNodeOne = arrayEdge[0]
            stringNodeTwo = arrayEdge[1]

            if not (stringNodeOne == stringNodeTwo):
                if stringNodeOne in listBackground:
                    numNodeOneIndex = listBackground.index(stringNodeOne)
                else:
                    numNodeOneIndex = -1

                if stringNodeTwo in listBackground:
                    numNodeTwoIndex = listBackground.index(stringNodeTwo)
                else:
                    numNodeTwoIndex = -1

                arrayNodeOneData = arrayProteinData[numNodeOneIndex,:]
                arrayNodeTwoData = arrayProteinData[numNodeTwoIndex,:]

                arrayOutDataOne = []
                arrayOutDataTwo = []
                for i in range(len(arrayNodeOneData)):
                    if ((arrayNodeOneData[i] > 0.) and (arrayNodeTwoData[i] > 0.)):
                        arrayOutDataOne.append(arrayNodeOneData[i])
                        arrayOutDataTwo.append(arrayNodeTwoData[i])

                if ((len(arrayOutDataOne) > 5) and (len(arrayOutDataTwo) > 5)):
                    arrayCorr = np.corrcoef([arrayOutDataOne, arrayOutDataTwo])
                    arrayEdgeCorr[iEdge] = arrayCorr[0,1]
                else:
                    arrayEdgeCorr[iEdge] = np.nan

            else:
                arrayEdgeCorr[iEdge] = np.nan

        arrayRandNetworkCorrs = np.zeros((graphNetwork.number_of_edges(),numPermTests), dtype=np.float_)
        numNodes = len(listBackground)
        for iPerm in range(numPermTests):
            arrayPermTestCorrs = np.zeros(graphNetwork.number_of_edges(), dtype=np.float_)

            for iEdge in range(graphNetwork.number_of_edges()):
                arrayRandNodeIndices = np.random.choice(range(numNodes), 2)

                stringNodeOne = listBackground[arrayRandNodeIndices[0]]
                stringNodeTwo = listBackground[arrayRandNodeIndices[1]]

                flagEdgeIsPPI = graphNetwork.has_edge(stringNodeOne, stringNodeTwo)
                if (flagSkipKnownPPIs and flagEdgeIsPPI):
                    arrayPermTestCorrs[iEdge] = np.nan
                else:
                    if not (arrayRandNodeIndices[0] == arrayRandNodeIndices[1]):

                        arrayNodeOneData = arrayProteinData[arrayRandNodeIndices[0],:]
                        arrayNodeTwoData = arrayProteinData[arrayRandNodeIndices[1],:]

                        arrayOutDataOne = []
                        arrayOutDataTwo = []
                        for i in range(len(arrayNodeOneData)):
                            if ((arrayNodeOneData[i] > 0.) and (arrayNodeTwoData[i] > 0.)):
                                arrayOutDataOne.append(arrayNodeOneData[i])
                                arrayOutDataTwo.append(arrayNodeTwoData[i])

                        if ((len(arrayOutDataOne) > 5) and (len(arrayOutDataTwo) > 5)):
                            arrayCorr = np.corrcoef([arrayOutDataOne, arrayOutDataTwo])
                            arrayPermTestCorrs[iEdge] = arrayCorr[0,1]
                        else:
                            arrayPermTestCorrs[iEdge] = np.nan

                    else:
                        arrayPermTestCorrs[iEdge] = np.nan

            arrayRandNetworkCorrs[:,iPerm] = arrayPermTestCorrs


        return {'arrayEdgeCorr':arrayEdgeCorr,
                'arrayRandNetworkCorrs':arrayRandNetworkCorrs}


#UniProt IDs are stored in structProteinLists['UniProtLists'][0] -> structProteinLists['UniProtLists'][numConditions-1]
#NB: the first one [0] contains the background network (all proteins detected across all cell lines)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# execute functions to load the data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# control data processing (i.e. load intermediate processed files to decrease run time)
flagPerformHochgrafeDataExtraction = True
stringHochgrafeDataFormat = 'Protein'
strCellLineOfInterest = 'MM231'

flagPerformPINA2Extraction = True

numPermTests = 100

# define the file system location of the input files
strDataPath = 'C:\\doc\\methods_in_proteomics'
strPINA2Path = 'C:\\db\\pina2'

# define the file system location of the output files
strOutputFolder = 'C:\\doc\\methods_in_proteomics'
# check that the folder exists, if not, create
if not os.path.exists(strOutputFolder):
    os.makedirs(strOutputFolder)

# extract the specified Hochgrafe data
structHochgrafeData = Extract.hochgrafe_supp_table_3(strDataPath, flagPerformHochgrafeDataExtraction)
structProteinLists = Extract.hochgrafe_lists(structHochgrafeData, strCellLineOfInterest)

structPINANetwork = Extract.pina2_mitab(strPINA2Path, flagPerformPINA2Extraction)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# execute data analysis functions and plot the output
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# calculate background network statistics
structBackgroundNetworkStats = Test.network_features(structPINANetwork,
                                                     structProteinLists['UniProtBackground'],
                                                     numPermTests)

# calculate conditional network statistics
structCondNetworkStats = Test.network_features(structPINANetwork,
                                               structProteinLists['UniProtListByCondition'],
                                               numPermTests)

# recalculate the average correlations while excluding known PPIs
flagIgnorePPIs = True
structDataCorrNoPPIs = Test.edge_correlation(structPINANetwork,
                                             structHochgrafeData,
                                             numPermTests,
                                             flagIgnorePPIs)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot the output figure
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

numAnnotationFontSize = 10
numTitleFontSize = 14
numMaxYTicks = 3
numMaxXTicks = 4


# create a multi-panel figure for the final output
handleFig, arrayAxesHandles = plt.subplots(2,4)
handleFig.set_size_inches(12, 8)


# # # # # #
# plot the background network statistics across the first row
#   plot the number of connected nodes in the first column --> arrayAxesHandles[0,0]
# # # # # #

# calculate the empirical p-value
if all(structBackgroundNetworkStats['arrayRandNetworkConnectedNodes'] < structBackgroundNetworkStats['numConnectedNodes']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structBackgroundNetworkStats['arrayRandNetworkConnectedNodes'] >=
                                   structBackgroundNetworkStats['numConnectedNodes']))
    numPVal = numAboveVal/numPermTests

# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(structBackgroundNetworkStats['arrayRandNetworkConnectedNodes']) - \
                           min(structBackgroundNetworkStats['arrayRandNetworkConnectedNodes'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,0].hist(structBackgroundNetworkStats['arrayRandNetworkConnectedNodes'],
                               np.int(numDataXRangeForHistBins),
                               facecolor='b', edgecolor='b',
                               alpha=0.75, color='b')
# draw a vertical line for the observed value
arrayAxesHandles[0,0].axvline(structBackgroundNetworkStats['numConnectedNodes'],
                              linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[0,0].annotate(('Connected nodes = ' + str(structBackgroundNetworkStats['numConnectedNodes']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structBackgroundNetworkStats['numConnectedNodes'],
                                   0.65*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(structBackgroundNetworkStats['numConnectedNodes']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))

# label the plot
arrayAxesHandles[0,0].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,0].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,0].yaxis.set_major_locator(arrayYTickLoc)

# label the y-axis in the first column plots
arrayAxesHandles[0,0].set_ylabel('Frequency', fontsize=numTitleFontSize)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), structBackgroundNetworkStats['numConnectedNodes']])
arrayAxesHandles[0,0].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,0].xaxis.set_major_locator(arrayXTickLoc)

# # # # # #
# plot the background network statistics across the first row
#   plot the diameter in the second column --> arrayAxesHandles[0,1]
# # # # # #

# calculate the empirical p-value
if all(structBackgroundNetworkStats['arrayRandNetworkDiameter'] < structBackgroundNetworkStats['numDiameter']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structBackgroundNetworkStats['arrayRandNetworkDiameter'] >=
                                   structBackgroundNetworkStats['numDiameter']))
    numPVal = numAboveVal/numPermTests

# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(structBackgroundNetworkStats['arrayRandNetworkDiameter']) - \
                           min(structBackgroundNetworkStats['arrayRandNetworkDiameter'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,1].hist(structBackgroundNetworkStats['arrayRandNetworkDiameter'],
                               np.int(numDataXRangeForHistBins), color='b')
# draw a vertical line for the observed value
arrayAxesHandles[0,1].axvline(structBackgroundNetworkStats['numDiameter'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[0,1].annotate(('Diameter = ' + str(structBackgroundNetworkStats['numDiameter']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structBackgroundNetworkStats['numDiameter'],
                                   0.95*np.float(arrayMaxYVal)),
                               xytext=(1.1*np.float(structBackgroundNetworkStats['numDiameter']),
                                       1.15*np.float(arrayMaxYVal)),
                               horizontalalignment='left',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))

# label the plot
arrayAxesHandles[0,1].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,1].set_ylim(0, arrayMaxYVal*1.30)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,1].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), structBackgroundNetworkStats['numDiameter']])
arrayAxesHandles[0,1].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,1].xaxis.set_major_locator(arrayXTickLoc)


# # # # # #
# plot the background network statistics across the first row
#   plot the average connectivity in the third column --> arrayAxesHandles[0,2]
# # # # # #


# calculate the empirical p-value
if all(structBackgroundNetworkStats['arrayRandNetworkAvgClustering'] <
               structBackgroundNetworkStats['numAvgClustering']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structBackgroundNetworkStats['arrayRandNetworkAvgClustering'] >=
                                   structBackgroundNetworkStats['numAvgClustering']))
    numPVal = numAboveVal/numPermTests
# the average clustering value is continuous so produce a histogram for the permutation test distribution with nbins
#  scaled by the number of permutation tests
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,2].hist(structBackgroundNetworkStats['arrayRandNetworkAvgClustering'],
                               np.int(numPermTests/5), color='b')
# draw a vertical line for the observed value
arrayAxesHandles[0,2].axvline(structBackgroundNetworkStats['numAvgClustering'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq[1:])
arrayAxesHandles[0,2].annotate(('Average clustering\ncoefficient = ' +
                                "{0:.3f}".format(structBackgroundNetworkStats['numAvgClustering']) +
                                ';\n p-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structBackgroundNetworkStats['numAvgClustering'],
                                   0.65*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(structBackgroundNetworkStats['numAvgClustering']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))
# label the plot
arrayAxesHandles[0,2].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,2].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,2].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), structBackgroundNetworkStats['numAvgClustering']])
if min(arrayHistBins) == 0:
    numMinXVal = 0
else:
    numMinXVal = 0.8*min(arrayHistBins)
arrayAxesHandles[0,2].set_xlim(numMinXVal, numMaxXVal*1.05)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,2].xaxis.set_major_locator(arrayXTickLoc)

# # # # # #
# plot the condition-specific (cell line specific) network statistics across the second row
#   plot the number of connected nodes in the first column --> arrayAxesHandles[1,0]
# # # # # #

# calculate the empirical p-value
if all(structCondNetworkStats['arrayRandNetworkConnectedNodes'] <
               structCondNetworkStats['numConnectedNodes']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structCondNetworkStats['arrayRandNetworkConnectedNodes'] >=
                                   structCondNetworkStats['numConnectedNodes']))
    numPVal = numAboveVal/numPermTests

# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(structCondNetworkStats['arrayRandNetworkConnectedNodes']) - \
                           min(structCondNetworkStats['arrayRandNetworkConnectedNodes'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,0].hist(structCondNetworkStats['arrayRandNetworkConnectedNodes'],
                               np.int(numDataXRangeForHistBins), color='b')
# draw a vertical line for the observed value
arrayAxesHandles[1,0].axvline(structCondNetworkStats['numConnectedNodes'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[1,0].annotate(('Connected nodes = ' + str(structCondNetworkStats['numConnectedNodes']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structCondNetworkStats['numConnectedNodes'],
                                   0.65*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(structCondNetworkStats['numConnectedNodes']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))

# label the plot
arrayAxesHandles[1,0].set_title((strCellLineOfInterest + ' network'),
                                fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[1,0].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[1,0].yaxis.set_major_locator(arrayYTickLoc)

# label the y-axis in the first column plots
arrayAxesHandles[1,0].set_ylabel('Frequency',
                                 fontsize=numTitleFontSize)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), structCondNetworkStats['numConnectedNodes']])
arrayAxesHandles[1,0].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[1,0].xaxis.set_major_locator(arrayXTickLoc)

# label the x-axis for plots on the second row
arrayAxesHandles[1,0].set_xlabel('Number of connected nodes\nin largest sub-network',
                                 fontsize=numAnnotationFontSize)


# # # # # #
# plot the condition-specific (cell line specific) network statistics across the second row
#   plot the diameter in the second column --> arrayAxesHandles[1,1]
# # # # # #

# calculate the empirical p-value
if all(structCondNetworkStats['arrayRandNetworkDiameter'] <
               structCondNetworkStats['numDiameter']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structCondNetworkStats['arrayRandNetworkDiameter'] >=
                                   structCondNetworkStats['numDiameter']))
    numPVal = numAboveVal/numPermTests

# extract the range of the diameter for producing the histogram
numDataXRangeForHistBins = max(structCondNetworkStats['arrayRandNetworkDiameter']) - \
                           min(structCondNetworkStats['arrayRandNetworkDiameter'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,1].hist(structCondNetworkStats['arrayRandNetworkDiameter'],
                               np.int(numDataXRangeForHistBins), color='b')

# draw a vertical line for the observed value
arrayAxesHandles[1,1].axvline(structCondNetworkStats['numDiameter'],
                              linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[1,1].annotate(('Diameter = ' + str(structCondNetworkStats['numDiameter']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structCondNetworkStats['numDiameter'],
                                   0.95*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(structCondNetworkStats['numDiameter']),
                                       1.15*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))

# label the plot
arrayAxesHandles[1,1].set_title((strCellLineOfInterest + ' network'),
                                fontsize=numTitleFontSize)

# scale and label the y-axis
arrayAxesHandles[1,1].set_ylim(0, arrayMaxYVal*1.30)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[1,1].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis
numMaxXVal = max([max(arrayHistBins), structCondNetworkStats['numDiameter']])
arrayAxesHandles[1,1].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[1,1].xaxis.set_major_locator(arrayXTickLoc)

# label the x-axis for plots on the second row
arrayAxesHandles[1,1].set_xlabel('Diameter of the\nlargest sub-network', fontsize=numTitleFontSize)

# # # # # #
# plot the condition-specific (cell line specific) network statistics across the second row
#   plot the average connectivity in the third column --> arrayAxesHandles[1,2]
# # # # # #

# calculate the empirical p-value
if all(structCondNetworkStats['arrayRandNetworkAvgClustering'] < structCondNetworkStats['numAvgClustering']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(structCondNetworkStats['arrayRandNetworkAvgClustering'] >= structCondNetworkStats['numAvgClustering']))
    numPVal = numAboveVal/numPermTests

# the average clustering value is continuous so produce a histogram for the permutation test distribution with nbins
#  scaled by the number of permutation tests
arrayAxesHandles[1,2].hist(structCondNetworkStats['arrayRandNetworkAvgClustering'],
                           np.int(numPermTests/5))

arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,2].hist(structCondNetworkStats['arrayRandNetworkAvgClustering'],
                               np.int(numPermTests/5),
                               color='b')
# draw a vertical line for the observed value
arrayAxesHandles[1,2].axvline(structCondNetworkStats['numAvgClustering'],
                              linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq[1:])
arrayAxesHandles[1,2].annotate(('Average clustering\ncoefficient = ' +
                                "{0:.3f}".format(structCondNetworkStats['numAvgClustering']) +
                                ';\n p-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(structCondNetworkStats['numAvgClustering'], 0.65*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(structCondNetworkStats['numAvgClustering']), 0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right', fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'))

# label the plot
arrayAxesHandles[1,2].set_title((strCellLineOfInterest + ' network'), fontsize=numTitleFontSize)

# scale and label the y-axis
arrayAxesHandles[1,2].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[1,2].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis
numMaxXVal = max([max(arrayHistBins), structCondNetworkStats['numAvgClustering']])
if min(arrayHistBins) == 0:
    numMinXVal = 0
else:
    numMinXVal = 0.8*min(arrayHistBins)
arrayAxesHandles[1,2].set_xlim(numMinXVal, numMaxXVal*1.05)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[1,2].xaxis.set_major_locator(arrayXTickLoc)

# label the x-axis for plots on the second row
arrayAxesHandles[1,2].set_xlabel('Average clustering\ncoefficient', fontsize=numTitleFontSize)


# # # # # #
# plot the average network correlation without discarding known PPIs in the second row
#   plot the average absolute correlation in the fourth column --> arrayAxesHandles[0,3]
# # # # # #

# calculate the average absolute correlation across the condition-specific network
arrayNanCorrFlag = np.isnan(structDataCorrNoPPIs['arrayEdgeCorr'])
arrayNotNanCorrIndexArrays = np.where(arrayNanCorrFlag == False)
arrayNotNanCorrIndices = arrayNotNanCorrIndexArrays[0]
numMeanAbsCorr = np.average(abs(structDataCorrNoPPIs['arrayEdgeCorr'][arrayNotNanCorrIndices]))

# calculate the average absolute correlation across each permutation-test network
arrayRandNetworkMeanAbsCorr = np.zeros(numPermTests, dtype=np.float_)
for iPermTest in range(numPermTests):
    arrayRandNetworkCorr = structDataCorrNoPPIs['arrayRandNetworkCorrs'][:,iPermTest]
    arrayNanCorrFlag = np.isnan(arrayRandNetworkCorr)
    arrayNotNanCorrIndices = np.where(arrayNanCorrFlag == False)
    numRandNetworkMeanAbsCorr = np.average(abs(arrayRandNetworkCorr[arrayNotNanCorrIndices]))
    arrayRandNetworkMeanAbsCorr[iPermTest] = numRandNetworkMeanAbsCorr

# calculate the empirical p-value
if all(arrayRandNetworkMeanAbsCorr < numMeanAbsCorr):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(arrayRandNetworkMeanAbsCorr >= numMeanAbsCorr)
    numPVal = numAboveVal/numPermTests

# the average clustering value is continuous so produce a histogram for the permutation test distribution with nbins
#  scaled by the number of permutation tests
arrayHistFreq, arrayHistBins, arrayHistPatches = arrayAxesHandles[0,3].hist(arrayRandNetworkMeanAbsCorr,
                                                                            np.int(numPermTests/5),
                                                                            color='b')

# extract the maximum frequency ignoring the first bins (lots of zeros)
arrayMaxYVal = max(arrayHistFreq)

# draw a vertical line for the observed value
arrayAxesHandles[0,3].axvline(numMeanAbsCorr, linewidth=3, color='r')
# label the line with the p-value

arrayAxesHandles[0,3].text(numMeanAbsCorr, 1.10*np.float(arrayMaxYVal),
                           ('Average absolute\ncorrelation = ' + "{0:.3f}".format(numMeanAbsCorr) +
                            ';\n p-value <= ' + "{0:.4f}".format(numPVal)),
                           horizontalalignment='center', fontsize=numAnnotationFontSize)
# set the title
arrayAxesHandles[0,3].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis
arrayAxesHandles[0,3].set_ylim(0, arrayMaxYVal*1.3)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,3].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis
numMaxXVal = max([max(arrayHistBins), numMeanAbsCorr])
if min(arrayHistBins) == 0:
    numMinXVal = 0
else:
    numMinXVal = 0.8*min(arrayHistBins)
arrayAxesHandles[0,3].set_xlim(numMinXVal, numMaxXVal*1.05)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,3].xaxis.set_major_locator(arrayXTickLoc)


# label the x-axis for plots on the second row
arrayAxesHandles[0,3].set_xlabel('Average absolute\nPearson''s correlation', fontsize=numTitleFontSize)

# # # # # #

#set the figure to have a tight layout
handleFig.tight_layout()

#output as a 300 dpi figure
handleFig.savefig(os.path.join(strOutputFolder,'CombinedHists.png'), dpi=300)

#close the figure window
plt.close(handleFig)