import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

__author__ = 'Joe Cursons'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
#   class Extract:
#       --> hochgrafe_lists:
#       --> hochgrafe_supp_table_3:
#       --> pina2_mitab:
#   class Build:
#       --> ppi_graph:
#   class Test
#       --> network_features:
#       --> edge_correlation:
#
# These functions are executed over lines:
#
#
# This script has a number of dependencies on python packages, and we would like to acknowledge the developers of these
#  packages:
#   - pandas        ::      http://pandas.pydata.org/
#   - numpy         ::      http://www.numpy.org/
#   - networkx      ::      http://networkx.github.io/
#   - matplotlib    ::      http://matplotlib.org/
# NB: for Windows users who wish to use a 64-bit environment, it is recommended to use a pre-compiled set of python
#       packages, such as those provided by:
#           - WinPython: http://winpython.github.io/
#   For experienced Windows users, you may be able to install from the pre-compiled binaries for individual packages,
#       kindly provided Christoph Gohlke
#           - http://www.lfd.uci.edu/~gohlke/pythonlibs/
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# This python script was written by:
#       Joe Cursons, Walter and Eliza Hall Institute
#           - cursons.j (at) wehi.edu.au
#       Melissa Davis, Walter and Eliza Hall Institute
#           - davis.m (at) wehi.edu.au
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extract: a set of functions that extract data from specific files used in this analysis
# # # # # # # # # # # # # # # # #


class Extract:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # hochgrafe_lists(): a function that reads in the Hochgrafe data and extracts lists of proteins detected (UniProt
    #                       IDs) for the specified cell lines
    # Inputs:
    #   - structInHochgrafeData:
    #       'CellLines': a list of strings referencing the cell lines from the Hochgrafe data set
    #       'UniProt' - a list of UniProt IDs for detected proteins within the Hockgrafe data
    #       'arrayProtAbund' - a 2D array (numUniProtIDs = numRows; numCellLines = numCols) containing peptide/protein
    #                           abundance data from Hochgrafe et al
    #   - strInCellLinesOfInt
    #       string containing the cell line of interest from the Hochgrafe data
    # Output:
    #   - a dictionary/structured array containing:
    #       'UniProtBackground' - a list of all proteins (UniProt ID) within the background (across all cell lines
    #                               examined) network
    #       'UniProtListByCondition' - a list of proteins (UniProt ID) detected within the specified condition (cell
    #                                   line) of interest
    # # # # # # # # # # # # # # # # #
    def hochgrafe_lists(structInHochgrafeData,
                        strInCellLine):

        # extract the cell line list from the Hochgrafe data dictionary
        listCellLines = structInHochgrafeData['CellLines']

        # for the specified cell line (strInCellLine), determine the index within the Hochgrafe data
        if strInCellLine in listCellLines:
            numCellLineIndex = listCellLines.index(strInCellLine)
        else:
            print('ERROR: ' + strInCellLine + ' is not present within the specified data, please use ' +
                  'a member of ' + listCellLines)

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
                'UniProtListByCondition':arrayCellLineUniProtRows}


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # hochgrafe_supp_table_3(): a function that specifically loads the protein data (stab_3.xsl) from Hochgrafe et. al
    #                           (2010) and extracts all listed fields
    # Inputs:
    #   - strInFolderPath: a string containing the os.path readable absolute folder path for stab_3.xls from
    #                       Hochgrafe et al.
    #   - flagPerformExtraction: a Boolean flag to control whether the data should be extracted, or whether an
    #                               intermediate saved file can be used
    # Output:
    #   - a dictionary/structured array containing:
    #       'HGNC' - a list of all proteins (HGNC symbol)
    #       'UniProt' - a list of all proteins (UniProt ID)
    #       'CellLines' - a list of cell lines
    #       'arrayProtAbund' - a 2D array (protein*cell line) containing the protein abundance data
    # # # # # # # # # # # # # # # # #
    def hochgrafe_supp_table_3(strInFolderPath, flagPerformExtraction):

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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # pina2_mitab(): a function that specifically loads the protein-protein interaction data from PINA v2.0 (through
    #                   the MI-TAB tsv) and exports all proteins (UniProt ID and corresponding protein name), together
    #                   with the connectivity matrix
    # Inputs:
    #   - strInFolderPath: a string containing the os.path readable absolute folder path for Homo sapiens-20140521.tsv
    #                       from Cowley et al.
    #   - flagPerformExtraction: a Boolean flag to control whether the data should be extracted, or whether an
    #                               intermediate saved file can be used
    # Output:
    #   - a dictionary/structured array containing:
    #       'HGNC' - a list of all proteins (HGNC symbol)
    #       'UniProt' - a list of all proteins (UniProt ID)
    #       'arrayIntNetwork' - a 2D connectivity matrix (protein*protein) containing the interaction data (edges)
    # # # # # # # # # # # # # # # # #
    def pina2_mitab(strInFolderPath, flagPerformExtraction):

        # set the input/output file names
        strDataFile = 'Homo sapiens-20140521.tsv'
        strOutputSaveFile = 'processedPINA2Data'

        # check that the pre-processed data file exists if flagPerformExtraction is set to false
        if np.bitwise_and((not flagPerformExtraction),
                          (not os.path.exists(os.path.join(strInFolderPath, (strOutputSaveFile + '.npz'))))):
            print('warning: flagPerformExtraction is set to False for Extract.pina2_mitab(), however ' +
                  'the processed data file (' + strOutputSaveFile + ') does not exist at the specified location;' +
                  'setting flagPerformExtraction = True, which may increase run time')
            flagPerformExtraction = True

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

        return {'HGNC':listOutputProteinHGNCs,
                'UniProt':listOutputUniProtIDs,
                'arrayIntNetwork':arrayInteractionNetwork}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Build: a set of functions that create networks of various types/with various properties
# # # # # # # # # # # # # # # # #


class Build:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ppi_graph(): a function that takes an input dict/structured array containing the PINA v2.0 PPI information, and
    #               a list of proteins for building a specific instance of a network
    #   - dictInPINANetwork: a dict containing the PINA v2.0 information as interactor lists and a connectivity matrix,
    #                           as output from Extract.pina2_mitab()
    #   - listProteinsInNetwork: a list of proteins (UniProt IDs) which are to be used for construction of the
    #                               specified network
    # Output:
    #   - graphOutputNetwork: a networkx undirected graph containing the network structure (protein-protein
    #                           interactions) fore the specified protein list
    # # # # # # # # # # # # # # # # #
    def ppi_graph(dictInPINANetwork, listProteinsInNetwork):

        # initialise a network graph for output
        graphOutputNetwork = nx.Graph()

        # populate the output network with the desired proteins
        graphOutputNetwork.add_nodes_from(listProteinsInNetwork)

        # step through every protein within the network
        for stringProtOne in listProteinsInNetwork:
            # look for interactions within the extracted PINA data
            if stringProtOne in dictInPINANetwork['UniProt']:
                # if they exist, identify the entry index
                numProtOneIndex = dictInPINANetwork['UniProt'].index(stringProtOne)
                # and use this to identify the indices for protein interaction partners
                arrayInteractionPartnerFlag = dictInPINANetwork['arrayIntNetwork'][numProtOneIndex, :]
                arrayInteractionPartnerIndices = np.where(arrayInteractionPartnerFlag)
                # step through all interaction partners
                for numProtTwoIndex in arrayInteractionPartnerIndices[0]:
                    # map from the index to the UniProt identifier
                    stringProtTwo = dictInPINANetwork['UniProt'][numProtTwoIndex]
                    if stringProtTwo in listProteinsInNetwork:
                        # create the corresponding edge within the protein-protein interaction network
                        graphOutputNetwork.add_edge(stringProtOne,stringProtTwo)

            else:
                # assume that this protein has no known PPIs, move on to the next protein in the list
                print('warning: ' + stringProtOne + ' can not be found within the protein-protein interaction data')

        # return the network graph
        return graphOutputNetwork

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Test: a set of functions that examine quantitative features associated with network structures using permutation
#       testing
# # # # # # # # # # # # # # # # #


class Test:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # network_features(): a function that takes a background network, components for a sub-network of interest, and
    #                       an integer specifying the number of permutation tests; and then quantifies a number of
    #                       network metrics within the sub-network of interest against the randomly-permuted networks.
    #   - dictInPINANetwork: a dict containing the PINA v2.0 information as interactor lists and a connectivity matrix,
    #                           as output from Extract.pina2_mitab()
    #   - listProteinsForSubnetwork: a list of proteins (UniProt IDs) which are to be used for construction of the
    #                                   sub-network of interest
    #   - numPermTests: an integer specifying the number of randomly-permuted network structures to examine for the
    #                       null distributions of calculated metrics
    # Output:
    #   - a dictionary/structured array containing:
    #       'arrayRandNetworkAvgClustering' - an numPermTests-length vector containing the average clustering
    #                                           coefficient for randomly permuted networks
    #       'arrayRandNetworkDiameter' - an numPermTests-length vector containing the diameter for the largest connected
    #                                       component (subgraph) within the specified sub-network
    #       'arrayRandNetworkConnectedNodes' - an numPermTests-length vector containing the number of nodes for the
    #                                           largest connected component (subgraph) within the specified sub-network
    #       'numAvgClustering' - the average clustering coefficient for the specified sub-network
    #       'numDiameter' - the diameter for the largest connected component within the specified sub-network
    #       'numConnectedNodes' - the number of nodes for the largest connected component within the specified
    #                               sub-network
    # # # # # # # # # # # # # # # # #
    def network_features(dictInPINANetwork, listProteinsForSubnetwork, numPermTests):

        # build the PPI graph/sub-network for specified nodes
        graphNetwork = Build.ppi_graph(dictInPINANetwork, listProteinsForSubnetwork)
        numNetworkNodes = len(listProteinsForSubnetwork)

        # calculate the average clustering coefficient for the defined sub-network
        numAvgClustering = nx.average_clustering(graphNetwork)

        # identify the largest connected subgraph within this network and extract this
        graphNetworkConnected = max(nx.connected_component_subgraphs(graphNetwork), key=len)

        # calculate the diameter and number of nodes within this largest connected sub-component
        numDiameter = nx.diameter(graphNetworkConnected)
        numConnectedNodes = nx.number_of_nodes(graphNetworkConnected)

        # create output vectors for network metrics from randomly permuted graph structures
        arrayRandNetworkAvgClustering = np.zeros(numPermTests,dtype=np.float_)
        arrayRandNetworkDiameter = np.zeros(numPermTests,dtype=np.int32)
        arrayRandNetworkConnectedNodes = np.zeros(numPermTests,dtype=np.int32)

        # perform the specified number of permutation tests
        for iPermTest in range(numPermTests):
            # randomly select a set of UniProt IDs which is of the same size as the original sub-network
            arrayRandUniProtIDs = np.random.choice(dictInPINANetwork['UniProt'], numNetworkNodes)
            # and build a graph structure from this
            graphRandNetworkOfSameSize = Build.ppi_graph(dictInPINANetwork, arrayRandUniProtIDs)
            # determine the average clustering coefficient for this random network
            numRandNetworkAvgClustering = nx.average_clustering(graphRandNetworkOfSameSize)

            # identify the largest connected sub-component within this graph
            graphRandNetworkOfSameSizeConnected = max(nx.connected_component_subgraphs(graphRandNetworkOfSameSize), key=len)

            # and calculate the diameter and number of nodes for this connected subgraph
            numRandNetworkDiameter = nx.diameter(graphRandNetworkOfSameSizeConnected)
            numRandNetworkConnectedNodes = nx.number_of_nodes(graphRandNetworkOfSameSizeConnected)

            # output to the appropriate arrays
            arrayRandNetworkAvgClustering[iPermTest] = numRandNetworkAvgClustering
            arrayRandNetworkDiameter[iPermTest] = numRandNetworkDiameter
            arrayRandNetworkConnectedNodes[iPermTest] = numRandNetworkConnectedNodes

            # output the permutation test status
            print('permutation test ' + str(iPermTest) + ' of ' + str(numPermTests) + ' completed')

        # return a dictionary containing the specified data under key/value pairs
        return {'arrayRandNetworkAvgClustering':arrayRandNetworkAvgClustering,
                'arrayRandNetworkDiameter':arrayRandNetworkDiameter,
                'arrayRandNetworkConnectedNodes':arrayRandNetworkConnectedNodes,
                'numAvgClustering':numAvgClustering,
                'numDiameter':numDiameter,
                'numConnectedNodes':numConnectedNodes}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # edge_correlation(): a function that takes the PINA v2.0 PPI data, together with an array of quantitative data for
    #                       individual nodes across the network, and calculates the correlation associated with every
    #                       edge with sufficient observations (hard-coded n > 5). The distribution of correlations is
    #                       then compared to the distribution calculated from randomly-permuted networks. A Boolean flag
    #                       controls whether known PPI edges are excluded from the randomly-permuted network structures.
    #   - dictPINANetwork: a dict containing the PINA v2.0 information as interactor lists and a connectivity matrix,
    #                       as output from Extract.pina2_mitab()
    #   - dictProteinData: a dict containing the protein abundance data for calculating correlations (across each edge),
    #                       as output from Extract.hochgrafe_supp_table_3(); i.e. containing a list of proteins (UniProt
    #                       IDs), a list of cell lines, and a (protein*cell line) array of (phospho-)protein abundance
    #   - numPermTests: an integer specifying the number of randomly-permuted network structures to examine for the
    #                       null distributions of correlations
    #   - flagSkipKnownPPIs: a Boolean flag to control whether PPIs present within the true data should be skipped when
    #                           calculating correlations across the random/permuted network
    # Output:
    #   - a dictionary/structured array containing:
    #       'arrayEdgeCorr' - an array containing the edge-wise correlations calculated across all known PPIs with more
    #                           than five data points present
    #       'arrayRandNetworkCorrs' - a 2D array containing nPerm sets of edge-wise correlations calculated across the
    #                                   randomly permuted network structures
    # # # # # # # # # # # # # # # # #
    def edge_correlation(dictPINANetwork, dictProteinData, numPermTests, flagSkipKnownPPIs):

        # the Hochgrafe data contain some 'multiple entry' proteins due to peptides with identity across multiple
        #  proteins, these 'shared peptide sequences' often map to proteins which form large signalling complexes and
        #  thus these entries are excluded to prevent excess influence upon network statistics (when included multiple
        #  times)
        listUniProtEntries = dictProteinData['UniProt']
        arrayListWithMultipleEntryFlag = np.zeros(len(listUniProtEntries),dtype=np.bool)
        for iEntry in range(len(listUniProtEntries)):
            if '/' in listUniProtEntries[iEntry]:
                arrayListWithMultipleEntryFlag[iEntry] = True
        arrayListWithSingleEntryIndices = np.where(arrayListWithMultipleEntryFlag == False)[0]

        # the background list is simply the full list without any 'multiple entry' components
        listBackground = [dictProteinData['UniProt'][i] for i in arrayListWithSingleEntryIndices]

        arrayAllProteinData = dictProteinData['arrayProtAbund']
        arrayProteinData = np.zeros((np.size(dictProteinData['arrayProtAbund'], 0),
                                     np.size(dictProteinData['arrayProtAbund'], 1)),
                                    dtype=np.float_)
        for iRow in range(len(arrayListWithSingleEntryIndices)):
            arrayProteinData[iRow,:] = arrayAllProteinData[arrayListWithSingleEntryIndices[iRow],:]

        # extract the corresponding network from the PINA2 data into a NetworkX graph
        graphNetwork = Build.ppi_graph(dictPINANetwork, listBackground)

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# In the remainder of this script, a number of settings are specified (as input strings/parameters) and then the
#  functions defined above are executed to produce Figure 3 within the associated textbook
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# specify all user-defined variables
# # # # # # # # # # # # # # # # #

# Boolean flags that control the execution of data extraction functions (i.e. load intermediate processed files to
#  decrease run time). Note that extraction of the Hochgrafe data is relatively quick, although if custom data sets are
#  used this may change. Extraction of the PINA2 data (and formatting as a connectivity matrix) can take some time, and
#  setting this to false will improve run time.
flagPerformHochgrafeDataExtraction = True
flagPerformPINA2Extraction = True

# a string specifying the cell-line of interest for generating a specific sub-network of interest, and calculating
#  associated network metrics for the worked example - here we examine the MDA-MB-231 (MM231) breast cancer cell line
# NB: this string must be part of the cell line set specified by Hochgrafe et al
strCellLineOfInterest = 'MM231'

# an integer specifying the number of permutation tests; as noted within the associated text, the number of permutation
#  tests required depends on the desired false discovery rate
numPermTests = 1000

# define the file system location of the input files
strDataPath = 'C:\\doc\\methods_in_proteomics'
# check that the folder exists, if not, generate an error
if not os.path.exists(strDataPath):
    print('ERROR: the specified data path (' + strDataPath + ') cannot be found, please modify strDataPath within' +
          '         the script, or create the folder and place the required data (Hochgrafe et al Table S3) within')

strPINA2Path = 'C:\\db\\pina2'
# check that the folder exists, if not, generate an error
if not os.path.exists(strPINA2Path):
    print('ERROR: the specified data path (' + strDataPath + ') cannot be found, please modify strPINA2Path within' +
          '         the script, or create the folder and place the required data (PINA v2.0 MITAB file) within')

# define the file system location of the output files
strOutputFolder = 'C:\\doc\\methods_in_proteomics'
# check that the folder exists, if not, create it
if not os.path.exists(strOutputFolder):
    os.makedirs(strOutputFolder)

# specify the size (inches) of the output figure
numFigOutWidth = 12
numFigOutHeight = 8

# specify the subplot formatting/layout for the output figure
numSubPlotRows = 2
numSubPlotCols = 4

# specify the font-sizes to use for the output plot annotation
numAnnotationFontSize = 10
numTitleFontSize = 14

# specify the maximum number of x- and y-axis ticks for the output sub-plots
numMaxYTicks = 3
numMaxXTicks = 4


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# extract the required data
# # # # # # # # # # # # # # # # #

# extract Table S3 from Hochgrafe et al for the (phospho-)protein abundance data
dictHochgrafeData = Extract.hochgrafe_supp_table_3(strDataPath, flagPerformHochgrafeDataExtraction)

# extract the full set of proteins detected across all experiments ('UniProtBackground'), and the set of proteins
#  detected within specified experiment (strCellLineOfInterest --> 'UniProtListByCondition')
dictProteinLists = Extract.hochgrafe_lists(dictHochgrafeData, strCellLineOfInterest)

# extract the PINA v2.0 protein-protein interaction data
dictPINANetwork = Extract.pina2_mitab(strPINA2Path, flagPerformPINA2Extraction)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# calculate network metrics for the background and condition-specific network and their corresponding randomly permuted
#  networks (of the same size)
# # # # # # # # # # # # # # # # #

# calculate background network statistics
dictBackgroundNetworkStats = Test.network_features(dictPINANetwork,
                                                   dictProteinLists['UniProtBackground'],
                                                   numPermTests)

# calculate conditional network statistics
dictCondNetworkStats = Test.network_features(dictPINANetwork,
                                             dictProteinLists['UniProtListByCondition'],
                                             numPermTests)

# calculate an empirical p-value for significance of the number of connected nodes (within the largest sub-network)
#  within the full/background network
if all(dictBackgroundNetworkStats['arrayRandNetworkConnectedNodes'] <
               dictBackgroundNetworkStats['numConnectedNodes']):
    # we can't have an empirical p-value of zero, we can only calculate the minimum value for the given number of
    #  permutation tests, and claim that it is lower than or equal to than this
    numBackgroundConnNodePVal = 1./np.float(numPermTests)
else:
    # the empirical p-value is bounded by the number of background/permuted network metrics which exceed the observed
    #  value
    numConnNodeAboveObsVal = np.size(np.where(dictBackgroundNetworkStats['arrayRandNetworkConnectedNodes'] >=
                                              dictBackgroundNetworkStats['numConnectedNodes']))
    # and the relative fraction of permuted values which exceed the observed metric
    numBackgroundConnNodePVal = np.float(numConnNodeAboveObsVal)/np.float(numPermTests)

# calculate an empirical p-value for significance of the network diameter (for the largest sub-network) within the
#   full/background network
if all(dictBackgroundNetworkStats['arrayRandNetworkDiameter'] <
               dictBackgroundNetworkStats['numDiameter']):
    # calculate the minimum value for the given number of permutation tests
    numBackgroundDiameterPVal = 1. / np.float(numPermTests)
else:
    numDiamAboveObsVal = np.size(np.where(dictBackgroundNetworkStats['arrayRandNetworkDiameter'] >=
                                          dictBackgroundNetworkStats['numDiameter']))
    numBackgroundDiameterPVal = np.float(numDiamAboveObsVal) / np.float(numPermTests)

# calculate an empirical p-value for significance of the average network clustering coefficient (for the largest
#   sub-network) within the full/background network
if all(dictBackgroundNetworkStats['arrayRandNetworkAvgClustering'] <
               dictBackgroundNetworkStats['numAvgClustering']):
    numBackgroundClusteringPVal = 1. / np.float(numPermTests)
else:
    numClusteringAboveObsVal = np.size(np.where(dictBackgroundNetworkStats['arrayRandNetworkAvgClustering'] >=
                                                dictBackgroundNetworkStats['numAvgClustering']))
    numBackgroundClusteringPVal = np.float(numClusteringAboveObsVal) / np.float(numPermTests)

# recalculate the average correlations while excluding known PPIs
flagIgnorePPIs = True
structDataCorrNoPPIs = Test.edge_correlation(dictPINANetwork,
                                             dictHochgrafeData,
                                             numPermTests,
                                             flagIgnorePPIs)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# create the output figure, plot the metrics for the specified/condition-specific network (MDA-MB-231) against the
#  corresponding values from the randomly-permuted network, and annotate the figure
# # # # # # # # # # # # # # # # #

# create a multi-panel figure for the final output
handleFig, arrayAxesHandles = plt.subplots(numSubPlotRows,numSubPlotCols)
handleFig.set_size_inches(w=numFigOutWidth, h=numFigOutHeight)

# # # # # # # # # # # # # # # # #
# plot background network statistics across the first row & number of connected nodes in the first column
#   --> arrayAxesHandles[0,0]
# # # # #
# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(dictBackgroundNetworkStats['arrayRandNetworkConnectedNodes']) - \
                           min(dictBackgroundNetworkStats['arrayRandNetworkConnectedNodes'])

# create the histogram for the permutation test distribution (in blue, with some transparency)
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,0].hist(dictBackgroundNetworkStats['arrayRandNetworkConnectedNodes'],
                               np.int(numDataXRangeForHistBins),
                               facecolor='b', edgecolor='b',
                               alpha=0.75, color='b')

# draw a vertical line for the observed value (in red)
arrayAxesHandles[0,0].axvline(dictBackgroundNetworkStats['numConnectedNodes'],
                              linewidth=3, color='r')

# label the line with the observed value, and the empirical p-value as a measure of significance
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[0,0].annotate(('Connected nodes = ' + str(dictBackgroundNetworkStats['numConnectedNodes']) +
                                ';\np-value <= ' + "{0:.4f}".format(numBackgroundConnNodePVal)),
                               xy=(dictBackgroundNetworkStats['numConnectedNodes'],
                                   0.65*np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(dictBackgroundNetworkStats['numConnectedNodes']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])

# label the plot
arrayAxesHandles[0,0].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,0].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,0].yaxis.set_major_locator(arrayYTickLoc)

# label the y-axis in the first column plots
arrayAxesHandles[0,0].set_ylabel('Frequency', fontsize=numTitleFontSize)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), dictBackgroundNetworkStats['numConnectedNodes']])
arrayAxesHandles[0,0].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,0].xaxis.set_major_locator(arrayXTickLoc)

# # # # # #
# plot the background network statistics across the first row
#   plot the diameter in the second column --> arrayAxesHandles[0,1]
# # # # # #

# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(dictBackgroundNetworkStats['arrayRandNetworkDiameter']) - \
                           min(dictBackgroundNetworkStats['arrayRandNetworkDiameter'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,1].hist(dictBackgroundNetworkStats['arrayRandNetworkDiameter'],
                               np.int(numDataXRangeForHistBins), color='b')
# draw a vertical line for the observed value
arrayAxesHandles[0,1].axvline(dictBackgroundNetworkStats['numDiameter'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[0,1].annotate(('Diameter = ' + str(dictBackgroundNetworkStats['numDiameter']) +
                                ';\np-value <= ' + "{0:.4f}".format(numBackgroundDiameterPVal)),
                               xy=(dictBackgroundNetworkStats['numDiameter'],
                                   0.95 * np.float(arrayMaxYVal)),
                               xytext=(1.1*np.float(dictBackgroundNetworkStats['numDiameter']),
                                       1.15*np.float(arrayMaxYVal)),
                               horizontalalignment='left',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])

# label the plot
arrayAxesHandles[0,1].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,1].set_ylim(0, arrayMaxYVal*1.30)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,1].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), dictBackgroundNetworkStats['numDiameter']])
arrayAxesHandles[0,1].set_xlim(min(arrayHistBins)-0.5, numMaxXVal+0.5)
arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
arrayAxesHandles[0,1].xaxis.set_major_locator(arrayXTickLoc)


# # # # # #
# plot the background network statistics across the first row
#   plot the average connectivity in the third column --> arrayAxesHandles[0,2]
# # # # # #

# the average clustering value is continuous so produce a histogram for the permutation test distribution with nbins
#  scaled by the number of permutation tests
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[0,2].hist(dictBackgroundNetworkStats['arrayRandNetworkAvgClustering'],
                               np.int(numPermTests/5), color='b', edgecolor='b')
# draw a vertical line for the observed value
arrayAxesHandles[0,2].axvline(dictBackgroundNetworkStats['numAvgClustering'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq[1:])
arrayAxesHandles[0,2].annotate(('Average clustering\ncoefficient = ' +
                                "{0:.3f}".format(dictBackgroundNetworkStats['numAvgClustering']) +
                                ';\n p-value <= ' + "{0:.4f}".format(numBackgroundClusteringPVal)),
                               xy=(dictBackgroundNetworkStats['numAvgClustering'],
                                   0.65 * np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(dictBackgroundNetworkStats['numAvgClustering']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])
# label the plot
arrayAxesHandles[0,2].set_title('Background network', fontsize=numTitleFontSize)

# scale and label the y-axis ticks
arrayAxesHandles[0,2].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[0,2].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis ticks
numMaxXVal = max([max(arrayHistBins), dictBackgroundNetworkStats['numAvgClustering']])
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
if all(dictCondNetworkStats['arrayRandNetworkConnectedNodes'] <
               dictCondNetworkStats['numConnectedNodes']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(dictCondNetworkStats['arrayRandNetworkConnectedNodes'] >=
                                   dictCondNetworkStats['numConnectedNodes']))
    numPVal = numAboveVal/numPermTests

# extract the range of the network connectivity for producing the histogram
numDataXRangeForHistBins = max(dictCondNetworkStats['arrayRandNetworkConnectedNodes']) - \
                           min(dictCondNetworkStats['arrayRandNetworkConnectedNodes'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,0].hist(dictCondNetworkStats['arrayRandNetworkConnectedNodes'],
                               np.int(numDataXRangeForHistBins), color='b', edgecolor='b')
# draw a vertical line for the observed value
arrayAxesHandles[1,0].axvline(dictCondNetworkStats['numConnectedNodes'], linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[1,0].annotate(('Connected nodes = ' + str(dictCondNetworkStats['numConnectedNodes']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(dictCondNetworkStats['numConnectedNodes'],
                                   0.65 * np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(dictCondNetworkStats['numConnectedNodes']),
                                       0.85*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])

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
numMaxXVal = max([max(arrayHistBins), dictCondNetworkStats['numConnectedNodes']])
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
if all(dictCondNetworkStats['arrayRandNetworkDiameter'] <
               dictCondNetworkStats['numDiameter']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(dictCondNetworkStats['arrayRandNetworkDiameter'] >=
                                   dictCondNetworkStats['numDiameter']))
    numPVal = numAboveVal/numPermTests

# extract the range of the diameter for producing the histogram
numDataXRangeForHistBins = max(dictCondNetworkStats['arrayRandNetworkDiameter']) - \
                           min(dictCondNetworkStats['arrayRandNetworkDiameter'])
# create the histogram for the permutation test distribution
arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,1].hist(dictCondNetworkStats['arrayRandNetworkDiameter'],
                               np.int(numDataXRangeForHistBins), color='b')

# draw a vertical line for the observed value
arrayAxesHandles[1,1].axvline(dictCondNetworkStats['numDiameter'],
                              linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq)
arrayAxesHandles[1,1].annotate(('Diameter = ' + str(dictCondNetworkStats['numDiameter']) +
                                ';\np-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(dictCondNetworkStats['numDiameter'],
                                   0.95 * np.float(arrayMaxYVal)),
                               xytext=(0.95*np.float(dictCondNetworkStats['numDiameter']),
                                       1.15*np.float(arrayMaxYVal)),
                               horizontalalignment='right',
                               fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])

# label the plot
arrayAxesHandles[1,1].set_title((strCellLineOfInterest + ' network'),
                                fontsize=numTitleFontSize)

# scale and label the y-axis
arrayAxesHandles[1,1].set_ylim(0, arrayMaxYVal*1.30)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[1,1].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis
numMaxXVal = max([max(arrayHistBins), dictCondNetworkStats['numDiameter']])
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
if all(dictCondNetworkStats['arrayRandNetworkAvgClustering'] < dictCondNetworkStats['numAvgClustering']):
    numPVal = 1/numPermTests
else:
    numAboveVal = np.size(np.where(dictCondNetworkStats['arrayRandNetworkAvgClustering'] >= dictCondNetworkStats['numAvgClustering']))
    numPVal = numAboveVal/numPermTests

# the average clustering value is continuous so produce a histogram for the permutation test distribution with nbins
#  scaled by the number of permutation tests
arrayAxesHandles[1,2].hist(dictCondNetworkStats['arrayRandNetworkAvgClustering'],
                           np.int(numPermTests/5))

arrayHistFreq, arrayHistBins, arrayHistPatches = \
    arrayAxesHandles[1,2].hist(dictCondNetworkStats['arrayRandNetworkAvgClustering'],
                               np.int(numPermTests/5),
                               color='b', edgecolor='b')
# draw a vertical line for the observed value
arrayAxesHandles[1,2].axvline(dictCondNetworkStats['numAvgClustering'],
                              linewidth=3, color='r')
# label the line with the p-value
arrayMaxYVal = max(arrayHistFreq[1:])
arrayAxesHandles[1,2].annotate(('Average clustering\ncoefficient = ' +
                                "{0:.3f}".format(dictCondNetworkStats['numAvgClustering']) +
                                ';\n p-value <= ' + "{0:.4f}".format(numPVal)),
                               xy=(dictCondNetworkStats['numAvgClustering'], 0.65 * np.float(arrayMaxYVal)),
                               xytext=(0.95 * np.float(dictCondNetworkStats['numAvgClustering']), 0.85 * np.float(arrayMaxYVal)),
                               horizontalalignment='right', fontsize=numAnnotationFontSize,
                               arrowprops=dict(facecolor='black'),
                               path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])

# label the plot
arrayAxesHandles[1,2].set_title((strCellLineOfInterest + ' network'), fontsize=numTitleFontSize)

# scale and label the y-axis
arrayAxesHandles[1,2].set_ylim(0, arrayMaxYVal*1.05)
arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
arrayAxesHandles[1,2].yaxis.set_major_locator(arrayYTickLoc)

# scale and label the x-axis
numMaxXVal = max([max(arrayHistBins), dictCondNetworkStats['numAvgClustering']])
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
                                                                            color='b', edgecolor='b')

# extract the maximum frequency ignoring the first bins (lots of zeros)
arrayMaxYVal = max(arrayHistFreq)

# draw a vertical line for the observed value
arrayAxesHandles[0,3].axvline(numMeanAbsCorr, linewidth=3, color='r')
# label the line with the p-value

arrayAxesHandles[0,3].text(numMeanAbsCorr, 1.10*np.float(arrayMaxYVal),
                           ('Average absolute\ncorrelation = ' + "{0:.3f}".format(numMeanAbsCorr) +
                            ';\n p-value <= ' + "{0:.4f}".format(numPVal)),
                           horizontalalignment='center', fontsize=numAnnotationFontSize,
                           path_effects=[path_effects.withStroke(linewidth=2,foreground="w")])
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