from itertools import chain
import numpy
import pickle
import pickle5 as pickle        # Liz/Mouhcine : ajout suite erreur protocole sur pickle

CONVERGENCE_LIMIT = 0.00000001

# Load the link information
with open("links.dict",'rb') as f:
	links = pickle.load(f)

# Remove redundant links (i.e. same link in the document)
for l in links:
	links[l] = list(set(links[l]))


# One click step in the "random surfer model"
def surfStep(origin, links):
	dest = [0.0] * len(origin)
	for idx, proba in enumerate(origin):
		if len(links[idx]):
			w = 1.0 / len(links[idx])     
			# Liz/Mouhcine : correction proba (loi uniforme) afin de simuler 
			# des random clicks ("random surfer model")
		else:
			w = 0.0
		for link in links[idx]:
			dest[link] += proba*w
	return dest



allPages = list(set().union(chain(*links.values()), links.keys()))
linksIdx = [ [allPages.index(target) for target in links.get(source,list())] for source in allPages ]


sourceVector = [1.0/len(allPages)] * len(allPages) 
# Liz/Mouhcine : initialiser une proba faible pour de potentielles pages isolées. 
# Ex : 10 pages sur le web : sourceVector = [0.1 , 0.1 , 0.1, ..., 0.1]  (vecteur de 10 valeurs)
pageRanks = [1.0/len(allPages)] * len(allPages)
delta = float("inf")

# Main loop for computing the Page Rank vector
while delta > CONVERGENCE_LIMIT:
	print("Convergence delta:",delta)
	pageRanksNew = surfStep(pageRanks, linksIdx)  # Liz/Mouhcine : analyse initiale du nb de liens entrants
	jumpProba = sum(pageRanks) - sum(pageRanksNew)
	if jumpProba < 0: # Technical artifact due to numerical float approximation
		jumpProba = 0
	# # Liz/Mouhcine : Add some source vector to avoid the SINK effect
	pageRanksNew = [ pageRank + jumpProba*jump for pageRank,jump in zip(pageRanksNew,sourceVector) ]
	# Liz/Mouhcine : (convergence implementation, question Q5.3)
	delta = sum(abs(b-a) for a,b in zip(pageRanksNew , pageRanks))
	pageRanks = pageRanksNew

# For information, what are the 10 highest ranked pages:
bestPages = reversed([ allPages[i] for i in numpy.argsort(pageRanks)[-10:] ])
bestPageRanks = reversed([ pageRanks[i] for i in numpy.argsort(pageRanks)[-10:] ])
for page,rank in zip(bestPages,bestPageRanks):
	print(page,"(rank score =",rank,")")


# Name the entries of the pageRank vector
pageRankDict = dict()
for idx,pageName in enumerate(allPages):
	pageRankDict[pageName] = pageRanks[idx]




# Save the ranks as pickle object
with open("pageRank.dict",'wb') as fileout:
	pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)



