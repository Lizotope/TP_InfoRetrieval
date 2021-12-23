from itertools import chain
import numpy
import pickle
import sys
import copy

import pickle5 as pickle				# Liz/Mouhcine : ajout suite erreur protocole sur pickle

with open("tokdoc.dict",'rb') as f:
	tokdoc = pickle.load(f)

with open("pageRank.dict",'rb') as f:
	pageRankDict = pickle.load(f)



Ntokens = sum(map(len,tokdoc.values()))     # Liz/Mouhcine : list of token 
docList = list(set(chain(*tokdoc.values())))   # Liz/Mouhcine : list of doc
Ndocs = len(docList)  # Liz/Mouhcine : total nb of doc


tokInfo = dict()
tf = dict()
tfidf = dict()

# Liz/Mouhcine : 
# tokdoc = analogie avec le cours = matrice où les lignes correspondent aux token, 
# et les colonnes correspondent aux documents. 
# Attention, ici le type n'est pas matrix mais dict parce que serialisation python !
# Question : si "thumb apparaît 4 fois dans le doc D, j'aurai 4 colonnes D?

# Compute the token information and count the token occurrences
for tok in tokdoc:   
# Liz/Mouhcine : pour chaque token tok de la matrice contenant les tok/docs

	tokInfo[tok] = -numpy.log(len(tokdoc[tok])/Ndocs)  
	# Liz/Mouhcine : verif : print("nb de docs total où est présent le token", tok, " : ", len(tokdoc[tok])) 
	# Liz/Mouhcine : tokInfo[tok]=idf du token tok = fréquence inverse de document 
	# = mesure de l'importance du terme tok dans l'ensemble du corpus'importance du terme dans l'ens du corpus
	# print("poids du token ", tok," dans le corpus ", tokInfo[tok]) 

	for doc in tokdoc[tok]:   
	# Liz/Mouhcine : pour chaque doc où est présent le token tok. 

		if not doc in tf:    # Liz/Mouhcine : si le document doc n'est pas présent dans le tf
			tf[doc] = dict() # Liz/Mouhcine : tf : alors je l'insère dans le dictionnaire tf 
			# Liz/Mouhcine : tf = term frequencies = en théorie, matrice où  ligne =docs et col = tokens, 
			# j'analyse doc par doc, et pour chaque doc, j'analyse token par token 
			# Attention, ici type python <>matrix mais un dictionnaire de dictionnaires ! 
		
		# Liz/Mouhcine : sur mon dictionnaire tf, pour chaque key doc, j'affecte comme valeur un dictionnaire 
		# dont la key est le token tok et dont la valeur est le nb d'occ rencontré à ce moment de tok dans doc
		tf[doc][tok] = tf[doc].get(tok,0) + 1  
		# Liz/Mouhcine : incrémentation du nb d'occurences du token tok dans le document doc. 
		# tf[doc][tok]= nb of occurence of the token tok in the document doc

# Normalize token occurrences to token frequencies
for doc in tf:
	Ntok = sum(tf[doc].values()) # Liz/Mouhcine : je somme les occurences de  tous les token pour le document doc
	for tok in tf[doc]:   
	# Liz/Mouhcine : pour chaque token tok du document doc
		tf[doc][tok] /= Ntok
		# Liz/Mouhcine : tf[doc][tok]= tf[doc][tok]/Ntok 
		# Liz/Mouhcine : verif : print ("pour le document ", doc, " la freq du token ", tok, " est ", tf[doc][tok])

# Compute the TF-IDF
for tok in tokdoc:
	for doc in tokdoc[tok]: 
	# Liz/Mouhcine : pr chq doc ou y'a le token tok
		if not doc in tfidf: 
		# Liz/Mouhcine : si le document doc n'est pas encore répertorié dans le tf-idf comme doc contenant le token tok
			tfidf[doc] = dict()  
			# Liz/Mouhcine : créer nouvel enregistrement où la clé est le document doc et la valeur est lui-même un 
			# dictionnaire dont la clé est le token tok et la valeur est le prod tf*idf
		# Liz/Mouhcine : tfidf[doc][tok] = calcul du produit tf * idf(=tokinfo)
		# Ce produit scalaire mesure le niveau de similarité/pertinence  entre le document doc et le token tok de la
		# query
		tfidf[doc][tok] = tf[doc][tok]*tokInfo[tok]

##############################################################################################################
# Liz/Mouhcine : le tf-idf est prêt une fois pour toutes, toute recherche sera maintenant "rapide"		
##############################################################################################################

# Liz/Mouhcine : : implementation de la fct de recherche sur la base d'une requête "query"
# Scalar product  
def scal(query,doc,tfidf):
	s = float()
	for tok in query:
		s += tfidf[doc].get(tok,0)*tokInfo[tok] 
		# Liz/Mouhcine : vérif : print("token : ", tok, "et doc ", doc) 
	return s

# Liz/Mouhcine : fct de récupération des topN 1ers résultats selon la pertinence des token
# Ranked by token relevance (vector model)
def getBestResults(queryStr, topN):
	query = queryStr.split(" ")
	searchRes = list(map(lambda d:scal(query,d,tfidf),docList))
	bestPages = list(reversed([ docList[i] for i in numpy.argsort(searchRes)[-topN:] ]))
	return bestPages

# Liz/Mouhcine : fct de récupération des rankings des pages de résultats 
# Page ranking of results
def rankResults(results):
	ranks = [ pageRankDict.get(page,0) for page in results ]
	rankedResults = list(reversed([ results[i] for i in numpy.argsort(ranks) ]))
	return rankedResults

# Liz/Mouhcine : fct d'impression des résultats top-és. 
def printResults(rankedResults):
	for idx,page in enumerate(rankedResults):
		print(str(idx) + ". " + page)

##############################################################################################################
#
# Liz/Mouhcine: Tests avec différentes query

query = "theory of evolution"
top = 15
print("results for query : ", query)
results = getBestResults(query,top)		# Liz/Mouhcine : récupération des résultats selon la pertinence des token
print("results by token relevance")
printResults(results)					# Liz/Mouhcine : impression des rés non "rankés" mais selon pertinence des token
rankedResults = rankResults(results)
print("results by rank")
printResults(rankedResults)				# Liz/Mouhcine : impression des résultats "rankés"

# Liz/Mouhcine : tests persos en isolant chaque mot de la query (au "stop word" "of" près)
# test 1
'''
query = "evolution"
top = 15
print("results for query : ", query)
results = getBestResults(query,top)
printResults(results)
rankedResults = rankResults(results)
printResults(rankedResults)
# test 2
query = "of"
top = 15
print("results for query : ", query)
results = getBestResults(query,top)
printResults(results)
rankedResults = rankResults(results)
printResults(rankedResults)
# test 3
query = "bacteria"
top = 15
print("results for query : ", query)
results = getBestResults(query,top)
printResults(results)
rankedResults = rankResults(results)
printResults(rankedResults)
'''