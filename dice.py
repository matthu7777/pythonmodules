import random, copy, numpy as np

class Die:
	def __init__(self, input):
		if (type(input) == int) or (type(input) == float):
			self.values = range(1,1+int(input))
		elif type(input) == list:
			self.values = input
		else:
			raise TypeError("Die input type %s not recognised"%(type(input)))
			
	def roll(self):
		val = random.choice(self.values)
		return val
	
	def __repr__(self):
		return repr(roll(self))
	
	def __neg__(self):
		self = copy.deepcopy(self)
		for j, val in enumerate(self.values):
			self.values[j] = -self.values[j]
		return self
	
	def __add__(a,b):
		a = copy.deepcopy(a)
		b = copy.deepcopy(b)
		if isinstance(a,Die) and isinstance(b,Die):
			return Dice([a,b])
		elif isinstance(a,Dice) and isinstance(b,Die):
			return Dice(a.diceList+[b], a.bonus)
		elif isinstance(a,Die) and isinstance(b,Dice):
			return Dice([a]+b.diceList, b.bonus)
		elif isinstance(a,Die) and (type(b)==int or type(b)==float):
			return Dice([a], b)
		elif isinstance(a,Dice) and (type(b)==int or type(b)==float):
			return Dice(a.diceList, b)
		else:
			raise TypeError("Inputs %s %s not recognised"%(type(a),type(b)))
	
	def __sub__(a,b):
		if isinstance(b,Die):
			b = copy.deepcopy(-b)
			return a+b
		elif (type(b)==int or type(b)==float):
			return Dice([a],b)
	
	def __mul__(a,b):
		if (type(b)==int or type(b)==float):
			if int(b) < 0:
				b = -b
				a = -a
			if int(b) == 0:
				return 0
			elif int(b) == 1:
				return copy.deepcopy(a)
			elif int(b) >= 2:
				dice = Dice([copy.deepcopy(a),copy.deepcopy(a)])
				if int(b) > 2:
					for j in xrange(2,b):
						dice = dice + copy.deepcopy(a)
				return dice
			else:
				raise TypeError("Input %f %s not recognised"%(b, type(b)))

	
	
class Dice:
	def __init__(self,diceList,bonus=0):
		self.diceList = diceList
		self.bonus = bonus
	
	def roll(self):
		total = self.bonus
		for die in self.diceList:
			val = die.roll()
			total += val
		return total
	
	def __repr__(self):
		return repr(roll(self))
	
	def __add__(a,b):
		a = copy.deepcopy(a)
		b = copy.deepcopy(b)
		if isinstance(a,Die) and isinstance(b,Die):
			return Dice([a,b])
		if isinstance(a,Dice) and isinstance(b,Die):
			return Dice(a.diceList+[b], a.bonus)
		if isinstance(a,Die) and isinstance(b,Dice):
			return Dice([a]+b.diceList, b.bonus)
		if isinstance(a,Die) and (type(b)==int or type(b)==float):
			return Dice([a], b)
		if isinstance(a,Dice) and (type(b)==int or type(b)==float):
			return Dice(a.diceList, b)
	
	def __sub__(a,b):
		if isinstance(b,Die):
			b = copy.deepcopy(-b)
			return a+b
		elif (type(b)==int or type(b)==float):
			return Dice([a],b)
	
	

def roll(a):
	return a.roll()
	
def attackroll(bonus,dc):
	return roll(d20+bonus) >= dc
	
def multiattack(bonus,dc,damageDice=None,number=1):
	hits = np.array([attackroll(bonus,dc) for j in range(number)]).sum()
	if damageDice:
		damage = np.array([damageDice.roll() for j in range(hits)]).sum()
		print "%d hits, %d damage"%(hits,damage)
	else:
		print "%d hits"%(hits)



d0 = Die(0)
d2 = Die(2)
d4 = Die(4)
d6 = Die(6)
d8 = Die(8)
d10 = Die(10)
d12 = Die(12)
d16 = Die(16)
d20 = Die(20)
d100 = Die(100)
coin = Die(["Heads","Tails"])

if __name__=="__main__":
	print "Rolling 2d20+3"
	print roll(d20*2 + 3)


