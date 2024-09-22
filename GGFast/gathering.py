from .core import Snippet, LVector

class Gathering:
    def __init__(self,n_gram_range = [1,8], top_k = 25000):
        self.n_gram_range = n_gram_range
        self.top_k = 25000
    
    def __getCandidatesFromVector(self,encoding : list, tp : int):

        def getSnippetsFromNGram(n_gram):
            snippets = []
            for i in range(0,len(encoding) - n_gram + 1):
                s1 = Snippet(sequence= encoding[i:i+n_gram], position= i+1, negation= False, tp= tp)
                s2 = Snippet(sequence= encoding[i:i+n_gram], position= -(len(encoding) - i), negation= False, tp= tp)
                s3 = Snippet(sequence= encoding[i:i+n_gram], position= "*", negation= False, tp= tp)

                snippets.append(s1)
                snippets.append(s2)
                snippets.append(s3)
            return snippets

        snippets = []
        for n_gram in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
            if len(encoding) < n_gram:
                break
            snippets.extend(getSnippetsFromNGram(n_gram= n_gram))
        return snippets


    def getCandidates(self,lvector : LVector):
        snippets = []
        snippets.extend(self.__getCandidatesFromVector(encoding= lvector.lv1, tp= 1))
        snippets.extend(self.__getCandidatesFromVector(encoding= lvector.lv2, tp= 2))
        snippets.extend(self.__getCandidatesFromVector(encoding= lvector.lv3, tp= 3))
        snippets.extend(self.__getCandidatesFromVector(encoding= lvector.lv4, tp= 4))
        snippets.extend(self.__getCandidatesFromVector(encoding= lvector.lv5, tp= 5))
        return snippets
        
