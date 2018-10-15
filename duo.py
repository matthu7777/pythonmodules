import duolingo, numpy as np
from sys import argv

d = duolingo.Duolingo('matthu7777')

if len(argv) > 1:
	lang = argv[1]
else:
	lang = 'French'

skills = d.get_learned_skills(d.get_abbreviation_of(lang))


strengths, names, pr = np.array([]),np.array([]),np.array([])
for skill in skills:
	strengths = np.append(strengths,skill[u'strength'])
	#np.append(pr,skill[u'practice_recommended'])
	names = np.append(names,skill[u'title'])


print "You have learnt", len(names), "skills"
print "Their average strength is", np.average(strengths)
print "You have", np.sum(strengths==strengths.min()), "skills at level", strengths.min()
if np.sum(strengths==strengths.min()) >= 3:
	print "Random weak skills:", np.random.choice(names[strengths==strengths.min()],3)
else:
	print "Weakest skills:", names[strengths==strengths.min()]
	

	

